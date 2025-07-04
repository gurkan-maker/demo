import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math
import base64
import tempfile
import os
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
import requests
from PIL import Image
import traceback
import kaleido

# ========================
# CONSTANTS & UNIT CONVERSION
# ========================
BAR_TO_KPA = 100  # 1 bar = 100 kPa
KPA_TO_BAR = 0.01
C_TO_K = 273.15   # Celsius to Kelvin
G_CONST = 9.80665  # Gravity constant (m/s²)

CONSTANTS = {
    "N1": {"gpm, psia": 1.00, "m³/h, bar": 0.865, "m³/h, kPa": 0.0865},
    "N2": {"mm": 0.00214, "inch": 890},
    "N5": {"mm": 0.00241, "inch": 1000},
    "N6": {"kg/h, kPa, kg/m³": 2.73, "kg/h, bar, kg/m³": 27.3, "lb/h, psia, lb/ft³": 63.3},
    "N7": {"m³/h, kPa, K (standard)": 4.17, "m³/h, bar, K (standard)": 417, "scfh, psia, R": 1360},
    "N8": {"kg/h, bar, K": 94.8, "lb/h, psia, R": 19.3},
    "N9": {"m³/h, kPa, K (standard)": 22.4, "m³/h, bar, K (standard)": 2240, "scfh, psia, R": 7320}
}

# Fluid properties (for simplified calculations)
WATER_DENSITY_4C = 999.97  # kg/m³
AIR_DENSITY_0C = 1.293     # kg/m³ at 0°C, 1 atm

# ========================
# FLUID LIBRARY
# ========================
FLUID_LIBRARY = {
    "Water": {
        "type": "liquid",
        "sg": 1.0,
        "visc": 1.0,  # cSt at 20°C
        "k": None,
        "pv_func": lambda t: calculate_vapor_pressure(t)
    },
    "Light Oil": {
        "type": "liquid",
        "sg": 0.85,
        "visc": 32.0,  # cSt at 40°C
        "k": None,
        "pv_func": lambda t: 0.0  # negligible vapor pressure
    },
    "Heavy Oil": {
        "type": "liquid",
        "sg": 0.92,
        "visc": 120.0,  # cSt at 40°C
        "k": None,
        "pv_func": lambda t: 0.0
    },
    "Air": {
        "type": "gas",
        "sg": 1.0,  # relative to air
        "visc": 0.0,  # not used for gas
        "k": 1.4,
        "pv_func": None
    },
    "Natural Gas": {
        "type": "gas",
        "sg": 0.6,  # relative to air
        "visc": 0.0,
        "k": 1.31,
        "pv_func": None
    },
    "Steam": {
        "type": "steam",
        "sg": None,  # not used for steam
        "visc": 0.0,
        "k": 1.33,
        "pv_func": None
    },
    "CO2": {
        "type": "gas",
        "sg": 1.52,
        "visc": 0.0,
        "k": 1.28,
        "pv_func": None
    },
    "Ammonia": {
        "type": "gas",
        "sg": 0.59,
        "visc": 0.0,
        "k": 1.32,
        "pv_func": None
    }
}

# ========================
# FLUID PROPERTY FUNCTIONS
# ========================
def calculate_vapor_pressure(temp_c: float) -> float:
    """Estimate vapor pressure for water using Antoine equation"""
    # Antoine constants for water (temp in °C, P in kPa)
    if temp_c < 0:
        A, B, C = 6.10649, 7.589, 240.71  # For ice
    else:
        A, B, C = 7.96681, 1668.21, 228.0  # For water (0-100°C)
    p_kpa = 10 ** (A - B/(temp_c + C))
    return p_kpa * KPA_TO_BAR  # Convert kPa to bar

def calculate_density(fluid: str, temp_c: float, press_bar: float) -> float:
    """Calculate fluid density (kg/m³) - simplified models"""
    t_k = temp_c + C_TO_K
    p_kpa = press_bar * BAR_TO_KPA
    
    if fluid == "water":
        # Water density approximation (more accurate near 4°C)
        base_density = 1000 - (temp_c - 4)**2 / 1600
        # Compressibility effect (simplified)
        return base_density * (1 + p_kpa * 5e-7)
    
    elif fluid == "air":
        # Ideal gas law: ρ = P * M / (R * T)
        return (p_kpa) * 28.97 / (8.314462 * t_k)  # kg/m³
    
    elif fluid == "steam":
        # Simplified saturated steam density
        # Actual calculation should use steam tables
        if press_bar < 0.1:
            return 0.01
        return 1 / (0.001 + 0.0002 * (200 - temp_c) + 0.00001 * press_bar**2)
    
    return 1000  # Default for unknown liquids

def calculate_kinematic_viscosity(fluid: str, temp_c: float) -> float:
    """Estimate kinematic viscosity in cSt (centistokes)"""
    if fluid == "water":
        # Water viscosity approximation (0-100°C)
        return 1.79 / (1 + 0.0337 * temp_c + 0.00022 * temp_c**2)
    elif fluid == "oil":
        # Typical hydraulic oil at 40°C is ~32 cSt
        return 32 * math.exp(-0.03 * (temp_c - 40))
    return 1.0  # Default for unknown fluids

# ========================
# VALVE DATABASE (UPDATED FORMAT)
# ========================
class Valve:
    def __init__(self, size_inch: int, rating_class: int, cv_table: dict, 
                 fl: float, xt_table: dict, fd_table: dict, d_inch: float = None,
                 valve_type: int = 3):  # 3=globe, 4=axial
        self.size = size_inch
        self.rating_class = rating_class
        self.cv_table = cv_table  # {open%: Cv}
        self.fl = fl  # Liquid pressure recovery factor
        self.xt = xt_table  # Pressure drop ratio factor
        self.fd = fd_table  # Valve style modifier (typically 0.7-1.0)
        self.diameter = d_inch if d_inch else size_inch * 0.95  # Approximate internal diameter
        self.valve_type = valve_type  # 3=globe, 4=axial
        
    def get_cv_at_opening(self, open_percent: float) -> float:
        """Get Cv at specified opening percentage with linear interpolation"""
        open_percent = max(10, min(100, open_percent))
        keys = sorted(self.cv_table.keys())
        
        # Find interpolation segment
        for i in range(len(keys)-1):
            if keys[i] <= open_percent <= keys[i+1]:
                x0, x1 = keys[i], keys[i+1]
                y0, y1 = self.cv_table[x0], self.cv_table[x1]
                return y0 + (y1 - y0) * (open_percent - x0) / (x1 - x0)
        
        # If beyond range, return min or max
        if open_percent <= keys[0]:
            return self.cv_table[keys[0]]
        return self.cv_table[keys[-1]]
    
# Comprehensive valve database with new naming format
VALVE_DATABASE = [
    # Globe valves (valve_type=3)
    Valve(0.5, 150, {10:1.2, 20:2.4, 30:4.0, 40:6.0, 50:8.5, 60:12, 70:16, 80:21, 90:26, 100:32}, 0.90, 0.72, 0.8, valve_type=3),
    Valve(1, 150, {10:3.0, 20:6.0, 30:10, 40:16, 50:24, 60:32, 70:42, 80:52, 90:62, 100:72}, 0.85, 0.75, 0.8, valve_type=3),
    Valve(1.5, 150, {10:6, 20:12, 30:20, 40:32, 50:48, 60:65, 70:85, 80:105, 90:125, 100:145}, 0.88, 0.70, 0.8, valve_type=3),
    Valve(2, 150, {10:10, 20:22, 30:36, 40:55, 50:80, 60:110, 70:140, 80:170, 90:200, 100:230}, 0.90, 0.68, 0.8, valve_type=3),
    Valve(3, 300, {10:25, 20:50, 30:80, 40:120, 50:170, 60:220, 70:280, 80:340, 90:400, 100:460}, 0.92, 0.65, 0.8, valve_type=3),
    Valve(4, 300, {10:45, 20:90, 30:140, 40:210, 50:300, 60:390, 70:490, 80:590, 90:690, 100:800}, 0.93, 0.62, 0.8, valve_type=3),
    Valve(1, 600, {10:8, 20:18, 30:30, 40:45, 50:65, 60:85, 70:110, 80:140, 90:170, 100:200}, 0.85, 0.70, 0.8, valve_type=3),
    Valve(2, 600, {10:20, 20:45, 30:75, 40:110, 50:155, 60:205, 70:265, 80:330, 90:400, 100:480}, 0.88, 0.65, 0.8, valve_type=3),
    Valve(3, 600, {10:45, 20:100, 30:170, 40:250, 50:350, 60:460, 70:590, 80:730, 90:880, 100:1050}, 0.90, 0.60, 0.8, valve_type=3),
    Valve(1, 1500, {10:15, 20:35, 30:60, 40:90, 50:130, 60:175, 70:230, 80:290, 90:360, 100:430}, 0.80, 0.75, 0.7, valve_type=3),
    Valve(2, 1500, {10:35, 20:80, 30:135, 40:200, 50:280, 60:370, 70:480, 80:600, 90:730, 100:880}, 0.82, 0.70, 0.7, valve_type=3),
    
    # Axial valves (valve_type=4)
    Valve(2, 300, {10:15, 20:35, 30:55, 40:80, 50:110, 60:150, 70:200, 80:250, 90:300, 100:350}, 0.92, 0.65, 0.9, valve_type=4),
    Valve(3, 300, {10:25, 20:60, 30:100, 40:150, 50:210, 60:280, 70:360, 80:450, 90:550, 100:650}, 0.94, 0.60, 0.9, valve_type=4),
    Valve(4, 300, {10:40, 20:90, 30:150, 40:220, 50:300, 60:400, 70:520, 80:650, 90:800, 100:950}, 0.95, 0.55, 0.9, valve_type=4),
    Valve(2, 900, {10:25, 20:55, 30:90, 40:130, 50:180, 60:240, 70:310, 80:390, 90:480, 100:580}, 0.93, 0.58, 0.9, valve_type=4),
    Valve(3, 900, {10:50, 20:115, 30:190, 40:280, 50:390, 60:520, 70:670, 80:840, 90:1020, 100:1220}, 0.95, 0.55, 0.9, valve_type=4),
    Valve(1, 2500, {10:20, 20:45, 30:75, 40:110, 50:155, 60:210, 70:275, 80:350, 90:435, 100:530}, 0.85, 0.68, 0.8, valve_type=4),
    Valve(2, 2500, {10:45, 20:100, 30:170, 40:250, 50:350, 60:470, 70:610, 80:770, 90:950, 100:1150}, 0.88, 0.62, 0.8, valve_type=4)
]

# ========================
# CV CALCULATION MODULE (ENHANCED)
# ========================
def reynolds_number(flow_m3h: float, d_m: float, visc_cst: float) -> float:
    """Calculate Reynolds number for viscosity correction"""
    # Re = 17,800 * Q / (ν * √Cv) - ISA equation
    # Where: Q = flow (m³/h), ν = kinematic viscosity (cSt)
    if visc_cst < 0.1:
        return 1e6  # Very low viscosity
    return 17800 * flow_m3h / (visc_cst * d_m * 1000)  # Approximated

def viscosity_correction(rev: float) -> float:
    """Calculate viscosity correction factor Fv per ISA-75.01.01"""
    if rev >= 10000:
        return 1.0
    elif rev <= 100:
        return 0.1 * rev ** 0.5
    return (1 + (rev / 7000) ** (1/0.7)) ** -0.7

def calculate_piping_factor_fp(valve_d: float, pipe_d: float) -> float:
    """Calculate piping geometry factor Fp for reducers/expanders"""
    # Simplified calculation per ISA-75.01.01
    if pipe_d <= valve_d:
        return 1.0
    ratio = valve_d / pipe_d
    return math.sqrt(1 + 0.5 * (1 - ratio**2))

def cv_liquid(flow: float, p1: float, p2: float, sg: float, fl: float, pv: float, 
              visc_cst: float, d_m: float, fp: float = 1.0) -> tuple:
    """
    Calculate Cv for liquids (ISA-75.01.01) with viscosity correction
    
    Args:
        flow: Volumetric flow (m³/h)
        p1: Inlet pressure (bar a)
        p2: Outlet pressure (bar a)
        sg: Specific gravity (water=1.0)
        fl: Liquid pressure recovery factor
        pv: Vapor pressure (bar a)
        visc_cst: Kinematic viscosity (cSt)
        d_m: Valve internal diameter (m)
        fp: Piping geometry factor
    
    Returns:
        (required_cv, theoretical_cv, details_dict)
    """
    dp = p1 - p2
    dp_max = fl**2 * (p1 - pv)  # Choking pressure drop
    
    # Theoretical Cv calculation (without corrections)
    if dp < dp_max:  # Non-choked flow
        theoretical_cv = (flow / N1) * math.sqrt(sg / dp)
    else:  # Choked flow
        theoretical_cv = (flow / N1) * math.sqrt(sg) / (fl * math.sqrt(p1 - ff * pv))
    
    # Apply piping factor
    cv_after_fp = theoretical_cv / fp
    
    # Viscosity correction
    rev = reynolds_number(flow, d_m, visc_cst)
    fr = viscosity_correction(rev)
    corrected_cv = cv_after_fp / fr
    
    # Return both theoretical and corrected Cv with details
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'fr': fr,
        'reynolds': rev,
        'is_choked': (dp >= dp_max)
    }
    
    return corrected_cv, details

def cv_gas(flow: float, p1: float, p2: float, sg: float, t: float, k: float, 
           xt: float, fp: float = 1.0) -> tuple:
    """
    Calculate Cv for gases (ISA-75.01.01)
    
    Args:
        flow: Standard flow rate (m³/h @ 15°C, 1 atm)
        p1: Inlet pressure (bar a)
        p2: Outlet pressure (bar a)
        sg: Specific gravity (air=1.0)
        t: Temperature (°C)
        k: Specific heat ratio (Cp/Cv)
        xt: Pressure drop ratio factor
        fp: Piping geometry factor
    
    Returns:
        (corrected_cv, details_dict)
    """
    t_k = t + C_TO_K
    x = (p1 - p2) / p1
    fk = k / 1.4  # Specific heat ratio factor
    is_choked = False
    
    if x >= fk * xt:  # Choked flow
        y = 0.667  # Expansion factor limit
        x = fk * xt
        is_choked = True
    else:  # Non-choked flow
        y = 1 - x / (3 * fk * xt)
    
    # Theoretical Cv without corrections
    theoretical_cv = (flow / N9) * math.sqrt((sg * t_k) / (x * p1**2)) / y
    corrected_cv = theoretical_cv / fp
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'expansion_factor': y,
        'is_choked': is_choked
    }
    
    return corrected_cv, details

def cv_steam(flow: float, p1: float, p2: float, rho: float, k: float, 
             xt: float, fp: float = 1.0) -> tuple:
    """
    Calculate Cv for steam (ISA-75.01.01)
    
    Args:
        flow: Mass flow rate (kg/h)
        p1: Inlet pressure (bar a)
        p2: Outlet pressure (bar a)
        rho: Density (kg/m³)
        k: Specific heat ratio (Cp/Cv)
        xt: Pressure drop ratio factor
        fp: Piping geometry factor
    
    Returns:
        (corrected_cv, details_dict)
    """
    x = (p1 - p2) / p1
    fk = k / 1.4
    is_choked = False
    
    if x >= fk * xt:  # Choked flow
        y = 0.667
        x = fk * xt
        is_choked = True
    else:  # Non-choked flow
        y = 1 - x / (3 * fk * xt)
    
    # Theoretical Cv without corrections
    theoretical_cv = flow / (N6 * y * math.sqrt(x * p1 * rho))
    corrected_cv = theoretical_cv / fp
    
    details = {
        'theoretical_cv': theoretical_cv,
        'fp': fp,
        'expansion_factor': y,
        'is_choked': is_choked
    }
    
    return corrected_cv, details

def check_cavitation(p1: float, p2: float, pv: float, fl: float) -> tuple:
    """
    Check for cavitation and choked flow conditions
    
    Returns:
        (is_choked, sigma, km) and cavitation warning message
    """
    dp = p1 - p2
    dp_max = fl**2 * (p1 - pv)
    km = fl**2  # Cavitation index
    
    # Calculate sigma (cavitation coefficient)
    sigma = (p1 - pv) / dp if dp > 0 else 1000
    
    if dp >= dp_max:
        return True, sigma, km, "Choked flow - cavitation likely"
    elif sigma < 1.5 * km:
        return False, sigma, km, "Severe cavitation risk"
    elif sigma < 2 * km:
        return False, sigma, km, "Moderate cavitation risk"
    return False, sigma, km, "Minimal cavitation risk"

# ========================
# PDF REPORT GENERATION
# ========================
class PDFReport(FPDF):
    def __init__(self, logo_bytes=None, logo_type=None):
        super().__init__()
        self.logo_bytes = logo_bytes
        self.logo_type = logo_type
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        # Logo
        if self.logo_bytes and self.logo_type:
            try:
                # Use temp file to avoid BytesIO issue
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{self.logo_type.lower()}") as tmpfile:
                    tmpfile.write(self.logo_bytes)
                    tmpfile_path = tmpfile.name
                
                self.image(tmpfile_path, x=10, y=8, w=30)
                os.unlink(tmpfile_path)  # Clean up temp file
            except Exception as e:
                self.cell(0, 10, f"Logo error: {str(e)}", 0, 1)
        
        # Title
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Control Valve Sizing Report', 0, 1, 'C')
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        # Line break
        self.ln(10)
        
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Arial', 'B', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)
        
    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
        
    def add_table(self, headers, data):
        # Calculate column widths
        col_widths = [40] * len(headers)  # Default width
        
        # Header
        self.set_font('Arial', 'B', 10)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, 1, 0, 'C')
        self.ln()
        
        # Data
        self.set_font('Arial', '', 10)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item), 1)
            self.ln()

def generate_pdf_report(scenarios, valve, op_points, req_cvs, warnings, cavitation_info, plot_bytes=None, logo_bytes=None, logo_type=None):
    """Generate a PDF report with sizing results"""
    pdf = PDFReport(logo_bytes=logo_bytes, logo_type=logo_type)
    pdf.add_page()
    
    # Report title and metadata
    pdf.chapter_title('Project Information')
    pdf.cell(0, 10, f'Project: Valve Sizing Analysis', 0, 1)
    pdf.cell(0, 10, f'Generated by: Valve Sizing Software', 0, 1)
    pdf.ln(5)
    
    # Selected valve information
    pdf.chapter_title('Selected Valve Details')
    valve_text = (
        f"Size: {valve.size}\" E{valve.valve_type}{valve.rating_class}\n"
        f"Type: {'Globe' if valve.valve_type == 3 else 'Axial'}\n"
        f"Rating Class: {valve.rating_class}\n"
        f"Fl (Liquid Recovery Factor): {valve.fl:.3f}\n"
        f"Xt (Pressure Drop Ratio): {valve.xt:.3f}\n"
        f"Fd (Valve Style Modifier): {valve.fd:.2f}\n"
        f"Internal Diameter: {valve.diameter:.2f} in"
    )
    pdf.chapter_body(valve_text)
    
    # Cv table
    pdf.chapter_title('Valve Cv Characteristics')
    cv_table_data = []
    for open_percent, cv in valve.cv_table.items():
        cv_table_data.append([f"{open_percent}%", f"{cv:.1f}"])
    pdf.add_table(['Opening %', 'Cv Value'], cv_table_data)
    
    # Sizing results for each scenario
    pdf.chapter_title('Sizing Results')
    results_data = []
    for i, scenario in enumerate(scenarios):
        actual_cv = valve.get_cv_at_opening(op_points[i])
        margin = (actual_cv / req_cvs[i] - 1) * 100 if req_cvs[i] > 0 else 0
        
        results_data.append([
            scenario["name"],
            f"{req_cvs[i]:.1f}",
            f"{valve.size}\"",
            f"{op_points[i]:.1f}%",
            f"{actual_cv:.1f}",
            f"{margin:.1f}%",
            warnings[i] + (" " + cavitation_info[i] if cavitation_info[i] else "")
        ])
    
    pdf.add_table(
        ['Scenario', 'Req Cv', 'Valve Size', 'Opening %', 'Actual Cv', 'Margin %', 'Warnings'],
        results_data
    )
    
    # Detailed calculation for each scenario
    pdf.chapter_title('Detailed Calculations')
    for i, scenario in enumerate(scenarios):
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, f'Scenario {i+1}: {scenario["name"]}', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        calc_text = (
            f"Fluid Type: {scenario['fluid_type'].title()}\n"
            f"Flow Rate: {scenario['flow']} "
            f"{'m³/h' if scenario['fluid_type']=='liquid' else 'kg/h' if scenario['fluid_type']=='steam' else 'std m³/h'}\n"
            f"Inlet Pressure (P1): {scenario['p1']:.2f} bar a\n"
            f"Outlet Pressure (P2): {scenario['p2']:.2f} bar a\n"
            f"Pressure Drop (dP): {scenario['p1'] - scenario['p2']:.2f} bar\n"
            f"Temperature: {scenario['temp']}°C\n"
        )
        
        if scenario["fluid_type"] == "liquid":
            calc_text += (
                f"Specific Gravity: {scenario['sg']:.3f}\n"
                f"Viscosity: {scenario['visc']} cSt\n"
                f"Vapor Pressure: {scenario['pv']:.4f} bar a\n"
                f"Cavitation Status: {cavitation_info[i]}\n"
            )
        elif scenario["fluid_type"] == "gas":
            calc_text += (
                f"Specific Gravity (air=1): {scenario['sg']:.3f}\n"
                f"Specific Heat Ratio (k): {scenario['k']:.3f}\n"
            )
        else:  # steam
            calc_text += (
                f"Density: {scenario['rho']:.3f} kg/m³\n"
                f"Specific Heat Ratio (k): {scenario['k']:.3f}\n"
            )
        
        calc_text += (
            f"Pipe Diameter: {scenario['pipe_d']} in\n"
            f"Required Cv: {req_cvs[i]:.1f}\n"
            f"Operating Point: {op_points[i]:.1f}% open\n"
            f"Actual Cv at Operating Point: {actual_cv:.1f}\n"
            f"Margin: {margin:.1f}%\n"
            f"Warnings: {warnings[i]}{', ' + cavitation_info[i] if cavitation_info[i] else ''}\n\n"
        )
        
        pdf.multi_cell(0, 5, calc_text)
    
    # Add the Cv curve plot if available
    if plot_bytes:
        pdf.chapter_title('Valve Cv Characteristic Curve')
        try:
            # Create temporary file for plot
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_plot:
                tmp_plot.write(plot_bytes)
                tmp_plot_path = tmp_plot.name
            
            # Pass file path to PDF
            pdf.image(tmp_plot_path, x=10, w=180)
            
            # Clean up temporary file
            os.unlink(tmp_plot_path)
        except Exception as e:
            pdf.cell(0, 10, f"Failed to insert plot: {str(e)}", 0, 1)
    
    # Save the PDF to a BytesIO object - FIXED
    pdf_string = pdf.output(dest='S').encode('latin1')
    pdf_bytes = BytesIO(pdf_string)
    pdf_bytes.seek(0)
    return pdf_bytes

# ========================
# SIMULATION RESULTS
# ========================
def get_simulation_image(valve_name):
    """Get simulation image URL based on valve name"""
    # Create a mapping of valve names to simulation images
    simulation_images = {
        "0.5\" E31": "https://via.placeholder.com/800x600.png?text=Simulation+0.5E31",
        "1\" E31": "https://via.placeholder.com/800x600.png?text=Simulation+1E31",
        "1.5\" E31": "https://via.placeholder.com/800x600.png?text=Simulation+1.5E31",
        "2\" E31": "https://via.placeholder.com/800x600.png?text=Simulation+2E31",
        "3\" E32": "https://via.placeholder.com/800x600.png?text=Simulation+3E32",
        "4\" E32": "https://via.placeholder.com/800x600.png?text=Simulation+4E32",
        "1\" E33": "https://via.placeholder.com/800x600.png?text=Simulation+1E33",
        "2\" E33": "https://via.placeholder.com/800x600.png?text=Simulation+2E33",
        "3\" E33": "https://via.placeholder.com/800x600.png?text=Simulation+3E33",
        "1\" E35": "https://via.placeholder.com/800x600.png?text=Simulation+1E35",
        "2\" E35": "https://via.placeholder.com/800x600.png?text=Simulation+2E35",
        "2\" E32": "https://via.placeholder.com/800x600.png?text=Simulation+2E32",
        "3\" E32": "https://via.placeholder.com/800x600.png?text=Simulation+3E32",
        "4\" E32": "https://via.placeholder.com/800x600.png?text=Simulation+4E32",
        "2\" E34": "https://via.placeholder.com/800x600.png?text=Simulation+2E34",
        "3\" E34": "https://via.placeholder.com/800x600.png?text=Simulation+3E34",
        "1\" E46": "https://via.placeholder.com/800x600.png?text=Simulation+1E46",
        "2\" E46": "https://via.placeholder.com/800x600.png?text=Simulation+2E46"
    }
    
    # Return the image URL or a placeholder if not found
    return simulation_images.get(valve_name, "https://via.placeholder.com/800x600.png?text=Simulation+Not+Available")

# ========================
# STREAMLIT APPLICATION
# ========================
def get_valve_display_name(valve):
    """Get formatted display name for valve"""
    # Create mapping from rating class to code
    rating_code_map = {
        150: 1,
        300: 2,
        600: 3,
        900: 4,
        1500: 5,
        2500: 6
    }
    
    # Get rating code (use class number if not in map)
    rating_code = rating_code_map.get(valve.rating_class, valve.rating_class)
    
    # Format: size" E(valve_type)(rating_code)
    return f"{valve.size}\" E{valve.valve_type}{rating_code}"

def create_valve_dropdown():
    """Create valve selection dropdown options"""
    # Add valves in sorted order
    valves = sorted(VALVE_DATABASE, key=lambda v: (v.size, v.rating_class, v.valve_type))
    valve_options = {get_valve_display_name(v): v for v in valves}
    return valve_options

def create_fluid_dropdown():
    """Create fluid selection dropdown options"""
    return ["Select Fluid Library..."] + list(FLUID_LIBRARY.keys())

def scenario_input_form(scenario_num, scenario_data=None):
    """Create input form for a scenario"""
    # Initialize all variables with default values
    default_values = {
        "sg": 1.0,
        "visc": 1.0,
        "pv": 0.023,
        "k": 1.4,
        "rho": 1.0,
        "fluid_type": "liquid"
    }
    
    # Initialize scenario_data with defaults if not provided
    if scenario_data is None:
        scenario_data = {
            "name": f"Scenario {scenario_num}",
            "fluid_type": "liquid",
            "flow": 10.0 if scenario_num == 1 else 50.0,
            "p1": 10.0,
            "p2": 6.0,
            "temp": 20.0,
            "pipe_d": 2.0
        }
        # Merge with default values
        scenario_data = {**default_values, **scenario_data}
    else:
        # Ensure we have all keys defined
        for key, default in default_values.items():
            if key not in scenario_data:
                scenario_data[key] = default
    
    st.subheader(f"Scenario {scenario_num}")
    
    # Name input
    scenario_name = st.text_input("Scenario Name", value=scenario_data["name"], key=f"name_{scenario_num}")
    
    # Top row: Fluid selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Fluid library selection
        fluid_library = st.selectbox(
            "Fluid Library", 
            create_fluid_dropdown(), 
            key=f"fluid_library_{scenario_num}"
        )
    
    with col2:
        # Fluid type selection - use try/except to prevent index errors
        try:
            index_val = ["Liquid", "Gas", "Steam"].index(scenario_data["fluid_type"].capitalize())
        except (ValueError, AttributeError):
            index_val = 0
        
        fluid_type = st.selectbox(
            "Fluid Type", 
            ["Liquid", "Gas", "Steam"], 
            index=index_val,
            key=f"fluid_type_{scenario_num}"
        ).lower()
    
    # Handle fluid library selection BEFORE other properties
    if fluid_library != "Select Fluid Library...":
        fluid_data = FLUID_LIBRARY[fluid_library]
        
        # Update fluid type in scenario data
        scenario_data["fluid_type"] = fluid_data["type"]
        
        # Update properties
        if fluid_data["sg"] is not None:
            scenario_data["sg"] = fluid_data["sg"]
        if fluid_data["visc"] is not None:
            scenario_data["visc"] = fluid_data["visc"]
        if fluid_data["k"] is not None:
            scenario_data["k"] = fluid_data["k"]
        
        # Calculate vapor pressure if applicable
        if fluid_data["pv_func"] and fluid_data["type"] == "liquid":
            scenario_data["pv"] = fluid_data["pv_func"](scenario_data["temp"])
        
        # Calculate density for steam
        if fluid_data["type"] == "steam":
            scenario_data["rho"] = calculate_density("steam", scenario_data["temp"], scenario_data["p1"])
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Flow rate
        flow_label = "Flow Rate (m³/h)" if fluid_type == "liquid" else "Flow Rate (std m³/h)" if fluid_type == "gas" else "Flow Rate (kg/h)"
        flow_value = st.number_input(
            flow_label, 
            min_value=0.0, 
            max_value=100000000000.0, 
            value=scenario_data["flow"], 
            step=0.1,
            key=f"flow_{scenario_num}"
        )
        
        # Pressure and temperature
        p1 = st.number_input(
            "Inlet Pressure (bar a)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=scenario_data["p1"], 
            step=0.1,
            key=f"p1_{scenario_num}"
        )
        
        p2 = st.number_input(
            "Outlet Pressure (bar a)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=scenario_data["p2"], 
            step=0.1,
            key=f"p2_{scenario_num}"
        )
        
        temp = st.number_input(
            "Temperature (°C)", 
            min_value=-200.0, 
            max_value=1000.0, 
            value=scenario_data["temp"], 
            step=1.0,
            key=f"temp_{scenario_num}"
        )
    
    with col2:
        # Fluid-specific properties
        if fluid_type in ["liquid", "gas"]:
            sg = st.number_input(
                "Specific Gravity (water=1)" if fluid_type == "liquid" else "Specific Gravity (air=1)",
                min_value=0.01, 
                max_value=10.0, 
                value=scenario_data["sg"], 
                step=0.01,
                key=f"sg_{scenario_num}"
            )
        
        if fluid_type == "liquid":
            visc = st.number_input(
                "Viscosity (cSt)", 
                min_value=0.01, 
                max_value=10000.0, 
                value=scenario_data["visc"], 
                step=0.1,
                key=f"visc_{scenario_num}"
            )
            
            pv_col, btn_col = st.columns([3, 1])
            with pv_col:
                pv = st.number_input(
                    "Vapor Pressure (bar a)", 
                    min_value=0.0, 
                    max_value=100.0, 
                    value=scenario_data["pv"], 
                    step=0.0001,
                    format="%.4f",
                    key=f"pv_{scenario_num}"
                )
            with btn_col:
                if st.button("Calculate", key=f"calc_pv_{scenario_num}"):
                    # Calculate and store in temporary key
                    st.session_state[f"temp_pv_{scenario_num}"] = calculate_vapor_pressure(temp)
                    st.rerun()
        
        if fluid_type in ["gas", "steam"]:
            k = st.number_input(
                "Specific Heat Ratio (k=Cp/Cv)", 
                min_value=1.0, 
                max_value=2.0, 
                value=scenario_data["k"], 
                step=0.01,
                key=f"k_{scenario_num}"
            )
        
        if fluid_type == "steam":
            density_col, btn_col = st.columns([3, 1])
            with density_col:
                rho = st.number_input(
                    "Density (kg/m³)", 
                    min_value=0.01, 
                    max_value=2000.0, 
                    value=scenario_data["rho"], 
                    step=0.1,
                    key=f"rho_{scenario_num}"
                )
            with btn_col:
                if st.button("Calculate", key=f"calc_rho_{scenario_num}"):
                    # Calculate and store in temporary key
                    st.session_state[f"temp_rho_{scenario_num}"] = calculate_density("steam", temp, p1)
                    st.rerun()
        
        pipe_d = st.number_input(
            "Pipe Diameter (inch)", 
            min_value=0.1, 
            max_value=100.0, 
            value=scenario_data["pipe_d"], 
            step=0.1,
            key=f"pipe_d_{scenario_num}"
        )
    
    # Handle temporary calculated values after rerun
    if f"temp_pv_{scenario_num}" in st.session_state:
        pv = st.session_state[f"temp_pv_{scenario_num}"]
        del st.session_state[f"temp_pv_{scenario_num}"]  # Clean up
    
    if f"temp_rho_{scenario_num}" in st.session_state:
        rho = st.session_state[f"temp_rho_{scenario_num}"]
        del st.session_state[f"temp_rho_{scenario_num}"]  # Clean up
    
    # Return updated scenario data
    return {
        "name": scenario_name,
        "fluid_type": fluid_type,
        "flow": flow_value,
        "p1": p1,
        "p2": p2,
        "temp": temp,
        "sg": sg if fluid_type in ["liquid", "gas"] else scenario_data["sg"],
        "visc": visc if fluid_type == "liquid" else scenario_data["visc"],
        "pv": pv if fluid_type == "liquid" else scenario_data["pv"],
        "k": k if fluid_type in ["gas", "steam"] else scenario_data["k"],
        "rho": rho if fluid_type == "steam" else scenario_data["rho"],
        "pipe_d": pipe_d
    }

def plot_cv_curve(valve, op_points, req_cvs, theoretical_cvs, scenario_names):
    """Plot the Cv curve of the selected valve with operating points"""
    # Generate Cv curve data
    openings = list(range(0, 101, 5))
    cv_values = [valve.get_cv_at_opening(op) for op in openings]
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Plot the curve
    fig.add_trace(go.Scatter(
        x=openings, 
        y=cv_values, 
        mode='lines',
        name='Valve Cv',
        line=dict(color='blue', width=3))
    )
    
    # Plot operating points
    for i, (op, req_cv, theoretical_cv) in enumerate(zip(op_points, req_cvs, theoretical_cvs)):
        actual_cv = valve.get_cv_at_opening(op)
        
        # Operating point marker
        fig.add_trace(go.Scatter(
            x=[op], 
            y=[actual_cv], 
            mode='markers+text',
            name=f'Scenario {i+1} Operating Point',
            marker=dict(size=12, color='red'),
            text=[f'S{i+1}'],
            textposition="top center"
        ))
        
        # Horizontal line for corrected Cv
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[req_cv, req_cv],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name=f'Corrected Cv S{i+1}',
            showlegend=False
        ))
        
        # Horizontal line for theoretical Cv
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[theoretical_cv, theoretical_cv],
            mode='lines',
            line=dict(color='green', dash='dot', width=1),
            name=f'Theoretical Cv S{i+1}',
            showlegend=False
        ))
    
    # Add annotations for required Cv values
    for i, (req_cv, theoretical_cv) in enumerate(zip(req_cvs, theoretical_cvs)):
        fig.add_annotation(
            x=100,
            y=req_cv,
            text=f'Corrected S{i+1}: {req_cv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=10,
            align='right',
            font=dict(color='red')
        )
        
        fig.add_annotation(
            x=100,
            y=theoretical_cv,
            text=f'Theoretical S{i+1}: {theoretical_cv:.1f}',
            showarrow=False,
            xshift=-10,
            yshift=-10,
            align='right',
            font=dict(color='green')
        )
    
    # Set layout
    fig.update_layout(
        title=f'{valve.size}" Valve Cv Characteristic',
        xaxis_title='Opening Percentage (%)',
        yaxis_title='Cv Value',
        legend_title='Legend',
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Set axis ranges
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, max(cv_values) * 1.1])
    
    return fig

def valve_3d_viewer(valve_name, model_url):
    """Display 3D valve model from a URL"""
    html_code = f"""
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <model-viewer src="{model_url}"
                  alt="{valve_name}"
                  auto-rotate
                  camera-controls
                  style="width: 100%; height: 500px;">
    </model-viewer>
    """
    components.html(html_code, height=520)

def main():
    # Configure page
    st.set_page_config(
        page_title="Control Valve Sizing",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS with larger font size
    st.markdown("""
        <style>
        html {
            font-size: 18px;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 15px 25px;
            border-radius: 10px 10px 0 0;
            font-size: 18px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        .stButton button {
            width: 100%;
            font-weight: bold;
            font-size: 18px;
        }
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-size: 18px;
        }
        .warning-card {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
        }
        .success-card {
            background-color: #d4edda;
            border-left: 5px solid #28a745;
        }
        .danger-card {
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
        }
        .logo-container {
            display: flex;
            justify-content: center;
            padding: 10px 0;
        }
        .simulation-modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            overflow: auto;
        }
        .modal-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 999;
        }
        .stMetric {
            font-size: 20px !important;
        }
        .stNumberInput, .stTextInput, .stSelectbox {
            font-size: 18px;
        }
        .stMarkdown {
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'valve' not in st.session_state:
        st.session_state.valve = None
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = None
    if 'logo_bytes' not in st.session_state:
        st.session_state.logo_bytes = None
    if 'logo_type' not in st.session_state:
        st.session_state.logo_type = None
    if 'show_simulation' not in st.session_state:
        st.session_state.show_simulation = False
    if 'show_3d_viewer' not in st.session_state:
        st.session_state.show_3d_viewer = False
    if 'cv_details' not in st.session_state:
        st.session_state.cv_details = {}
    
    # App title
    col1, col2 = st.columns([1, 4])
    with col1:
        # Check if logo exists in current directory
        default_logo = "logo.png"
        if os.path.exists(default_logo):
            st.image(default_logo, width=100)
        else:
            st.image("https://via.placeholder.com/100x100?text=LOGO", width=100)
    with col2:
        st.title("Control Valve Sizing Program")
        st.markdown("**ISA/IEC Standards Compliant Valve Sizing with Enhanced Visualization**")
    
    # Sidebar for valve selection and actions
    with st.sidebar:
        # Logo uploader
        st.header("VASTAŞ Logo")
        logo_upload = st.file_uploader("Upload VASTAŞ logo", type=["png", "jpg", "jpeg"], key="logo_uploader")
        
        if logo_upload is not None:
            # Store logo bytes and type
            st.session_state.logo_bytes = logo_upload.getvalue()
            st.session_state.logo_type = "PNG"  # Assume PNG for simplicity
            st.success("Logo uploaded successfully!")
        
        # Display logo in sidebar
        if st.session_state.logo_bytes:
            st.image(Image.open(BytesIO(st.session_state.logo_bytes)), use_container_width=True)
        elif os.path.exists("logo.png"):
            st.image(Image.open("logo.png"), use_container_width=True)
        else:
            st.image("https://via.placeholder.com/300x100?text=VASTAŞ+Logo", use_container_width=True)
        
        st.header("Valve Selection")
        
        # Valve selection dropdown
        valve_options = create_valve_dropdown()
        selected_valve_name = st.selectbox("Select Valve", list(valve_options.keys()))
        selected_valve = valve_options[selected_valve_name]
        
        # Action buttons
        st.header("Actions")
        calculate_btn = st.button("Calculate Opening", type="primary", use_container_width=True)
        export_btn = st.button("Export PDF Report", use_container_width=True)
        view_3d_btn = st.button("View 3D Model", use_container_width=True)
        show_simulation_btn = st.button("Show Simulation Results", use_container_width=True)
        
        # Valve details
        st.header("Valve Details")
        st.markdown(f"**Size:** {selected_valve.size}\"")
        st.markdown(f"**Type:** {'Globe' if selected_valve.valve_type == 3 else 'Axial'}")
        st.markdown(f"**Rating Class:** {selected_valve.rating_class}")
        st.markdown(f"**Fl (Liquid Recovery):** {selected_valve.fl:.3f}")
        st.markdown(f"**Xt (Pressure Drop Ratio):** {selected_valve.xt:.3f}")
        st.markdown(f"**Fd (Style Modifier):** {selected_valve.fd:.2f}")
        st.markdown(f"**Internal Diameter:** {selected_valve.diameter:.2f} in")
        
        # Cv table preview
        st.subheader("Cv Characteristics")
        cv_data = {"Opening %": list(selected_valve.cv_table.keys()), "Cv": list(selected_valve.cv_table.values())}
        cv_df = pd.DataFrame(cv_data)
        st.dataframe(cv_df, hide_index=True, height=300)
    
    # Handle button actions
    if view_3d_btn:
        st.session_state.show_3d_viewer = True
        st.session_state.show_simulation = False
    
    if show_simulation_btn:
        st.session_state.show_simulation = True
        st.session_state.show_3d_viewer = False
    
    # Main content tabs
    tab1, tab2, tab3, tab_results = st.tabs(["Scenario 1", "Scenario 2", "Scenario 3", "Results"])
    
    # Scenario 1
    with tab1:
        scenario1 = scenario_input_form(1)
    
    # Scenario 2
    with tab2:
        scenario2 = scenario_input_form(2)
    
    # Scenario 3
    with tab3:
        scenario3 = scenario_input_form(3)
    
    # Collect scenarios with flow > 0
    scenarios = []
    if scenario1["flow"] > 0:
        scenarios.append(scenario1)
    if scenario2["flow"] > 0:
        scenarios.append(scenario2)
    if scenario3["flow"] > 0:
        scenarios.append(scenario3)
    
    # Store scenarios in session state
    st.session_state.scenarios = scenarios
    
    # Calculate button action
    if calculate_btn:
        if not scenarios:
            st.error("Please define at least one scenario with flow > 0.")
            st.stop()
        
        # Calculate opening for each scenario
        operating_points = []
        required_cvs = []
        theoretical_cvs = []
        warnings = []
        cavitation_info = []
        cv_details = []
        
        for scenario in scenarios:
            # Get pipe diameter and calculate Fp
            pipe_d = scenario["pipe_d"]
            fp = calculate_piping_factor_fp(selected_valve.diameter, pipe_d)
            
            # Calculate required Cv
            if scenario["fluid_type"] == "liquid":
                # Calculate viscosity at temperature
                if scenario.get('fluid_library') == "Water":
                    scenario["visc"] = calculate_kinematic_viscosity("water", scenario["temp"])
                elif scenario.get('fluid_library') in ["Light Oil", "Heavy Oil"]:
                    scenario["visc"] = calculate_kinematic_viscosity("oil", scenario["temp"])
                
                cv_req, details = cv_liquid(
                    flow=scenario["flow"],
                    p1=scenario["p1"],
                    p2=scenario["p2"],
                    sg=scenario["sg"],
                    fl=selected_valve.fl,
                    pv=scenario["pv"],
                    visc_cst=scenario["visc"],
                    d_m=selected_valve.diameter * 0.0254,  # Convert inches to meters
                    fp=fp
                )
                
                # Store theoretical Cv
                theoretical_cvs.append(details['theoretical_cv'])
                
                # Check for cavitation
                choked, sigma, km, cav_msg = check_cavitation(
                    scenario["p1"], scenario["p2"], scenario["pv"], selected_valve.fl
                )
                cavitation_info.append(cav_msg)
                
                # Add to details
                details['sigma'] = sigma
                details['km'] = km
                cv_details.append(details)
                
            elif scenario["fluid_type"] == "gas":
                # Calculate density at temperature and pressure
                if scenario.get('fluid_library') == "Air":
                    scenario["sg"] = calculate_density("air", scenario["temp"], scenario["p1"]) / AIR_DENSITY_0C
                
                cv_req, details = cv_gas(
                    flow=scenario["flow"],
                    p1=scenario["p1"],
                    p2=scenario["p2"],
                    sg=scenario["sg"],
                    t=scenario["temp"],
                    k=scenario["k"],
                    xt=selected_valve.xt,
                    fp=fp
                )
                theoretical_cvs.append(details['theoretical_cv'])
                cavitation_info.append("N/A for gas")
                cv_details.append(details)
                
            else:  # steam
                # Calculate density if not provided
                if scenario["rho"] <= 0.1:
                    scenario["rho"] = calculate_density("steam", scenario["temp"], scenario["p1"])
                
                cv_req, details = cv_steam(
                    flow=scenario["flow"],
                    p1=scenario["p1"],
                    p2=scenario["p2"],
                    rho=scenario["rho"],
                    k=scenario["k"],
                    xt=selected_valve.xt,
                    fp=fp
                )
                theoretical_cvs.append(details['theoretical_cv'])
                cavitation_info.append("N/A for steam")
                cv_details.append(details)
            
            required_cvs.append(cv_req)
            
            # Find operating point
            open_percent = 10
            while open_percent <= 100:
                cv_valve = selected_valve.get_cv_at_opening(open_percent)
                if cv_valve >= cv_req:
                    break
                open_percent += 5  # Smaller step for precision
            
            operating_points.append(open_percent)
            
            # Check if within acceptable range
            warn = ""
            if open_percent < 20:
                warn = "Low opening (<20%)"
            elif open_percent > 80:
                warn = "High opening (>80%)"
            warnings.append(warn)
        
        # Store results
        st.session_state.results = {
            "valve": selected_valve,
            "op_points": operating_points,
            "req_cvs": required_cvs,
            "theoretical_cvs": theoretical_cvs,
            "warnings": warnings,
            "cavitation_info": cavitation_info,
            "scenario_names": [s["name"] for s in scenarios],
            "cv_details": cv_details
        }
    
    # Results tab
    with tab_results:
        if st.session_state.results:
            results = st.session_state.results
            valve = results["valve"]
            
            # Plot the Cv curve
            st.subheader("Valve Cv Characteristic")
            fig = plot_cv_curve(
                valve, 
                results["op_points"], 
                results["req_cvs"],
                results["theoretical_cvs"],
                results["scenario_names"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Sizing Results")
            results_data = []
            for i, scenario in enumerate(scenarios):
                actual_cv = valve.get_cv_at_opening(results["op_points"][i])
                margin = (actual_cv / results["req_cvs"][i] - 1) * 100 if results["req_cvs"][i] > 0 else 0
                
                # Determine status card
                status = "success-card" if 20 <= results["op_points"][i] <= 80 else "warning-card"
                if margin < 0:
                    status = "danger-card"
                
                # Warning messages
                warn_msgs = []
                if results["warnings"][i]:
                    warn_msgs.append(results["warnings"][i])
                if results["cavitation_info"][i]:
                    warn_msgs.append(results["cavitation_info"][i])
                warn_text = ", ".join(warn_msgs)
                
                with st.container():
                    st.markdown(f"<div class='result-card {status}'>", unsafe_allow_html=True)
                    
                    cols = st.columns([1.8, 1, 1, 1, 1, 1, 1, 1.5])
                    cols[0].markdown(f"**{scenario['name']}**")
                    cols[1].metric("Req Cv", f"{results['req_cvs'][i]:.1f}")
                    cols[2].metric("Theo Cv", f"{results['theoretical_cvs'][i]:.2f}")
                    cols[3].metric("Valve Cv", f"{valve.get_cv_at_opening(results['op_points'][i]):.1f}")
                    cols[4].metric("Valve Size", f"{valve.size}\"")
                    cols[5].metric("Opening", f"{results['op_points'][i]:.1f}%")
                    cols[6].metric("Margin", f"{margin:.1f}%", 
                                  delta_color="inverse" if margin < 0 else "normal")
                    cols[7].markdown(f"**{warn_text}**")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Detailed results
            st.subheader("Detailed Results")
            for i, scenario in enumerate(scenarios):
                with st.expander(f"Scenario {i+1}: {scenario['name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Process Conditions**")
                        st.markdown(f"- Fluid Type: {scenario['fluid_type'].title()}")
                        st.markdown(f"- Flow Rate: {scenario['flow']} "
                                    f"{'m³/h' if scenario['fluid_type']=='liquid' else 'kg/h' if scenario['fluid_type']=='steam' else 'std m³/h'}")
                        st.markdown(f"- Inlet Pressure (P1): {scenario['p1']:.2f} bar a")
                        st.markdown(f"- Outlet Pressure (P2): {scenario['p2']:.2f} bar a")
                        st.markdown(f"- Pressure Drop (dP): {scenario['p1'] - scenario['p2']:.2f} bar")
                        st.markdown(f"- Temperature: {scenario['temp']}°C")
                        
                    with col2:
                        st.markdown("**Fluid Properties**")
                        if scenario["fluid_type"] == "liquid":
                            st.markdown(f"- Specific Gravity: {scenario['sg']:.3f}")
                            st.markdown(f"- Viscosity: {scenario['visc']} cSt")
                            st.markdown(f"- Vapor Pressure: {scenario['pv']:.4f} bar a")
                        elif scenario["fluid_type"] == "gas":
                            st.markdown(f"- Specific Gravity (air=1): {scenario['sg']:.3f}")
                            st.markdown(f"- Specific Heat Ratio (k): {scenario['k']:.3f}")
                        else:  # steam
                            st.markdown(f"- Density: {scenario['rho']:.3f} kg/m³")
                            st.markdown(f"- Specific Heat Ratio (k): {scenario['k']:.3f}")
                        
                        st.markdown(f"- Pipe Diameter: {scenario['pipe_d']} in")
                    
                    st.markdown("**Sizing Results**")
                    st.markdown(f"- Theoretical Cv: {results['theoretical_cvs'][i]:.1f}")
                    st.markdown(f"- Corrected Cv: {results['req_cvs'][i]:.1f}")
                    st.markdown(f"- Operating Point: {results['op_points'][i]:.1f}% open")
                    st.markdown(f"- Actual Cv at Operating Point: {valve.get_cv_at_opening(results['op_points'][i]):.1f}")
                    
                    margin = ((valve.get_cv_at_opening(results['op_points'][i]) / results['req_cvs'][i] - 1) )* 100
                    st.markdown(f"- Margin: {margin:.1f}%")
                    
                    # Show correction factors
                    details = results['cv_details'][i]
                    st.markdown("**Correction Factors**")
                    if scenario["fluid_type"] == "liquid":
                        st.markdown(f"- Piping Factor (Fp): {details['fp']:.3f}")
                        st.markdown(f"- Viscosity Factor (Fr): {details['fr']:.3f}")
                        st.markdown(f"- Reynolds Number: {details['reynolds']:.0f}")
                        st.markdown(f"- Cavitation Sigma: {details['sigma']:.2f}")
                    else:
                        st.markdown(f"- Piping Factor (Fp): {details['fp']:.3f}")
                        st.markdown(f"- Expansion Factor (Y): {details['expansion_factor']:.3f}")
                    
                    # Warnings
                    if results['warnings'][i] or "risk" in results['cavitation_info'][i]:
                        st.warning(f"⚠️ {results['warnings'][i]}, {results['cavitation_info'][i]}")
        else:
            st.info("Click 'Calculate Opening' in the sidebar to see results")
        
        # Show 3D viewer if requested
        if st.session_state.show_3d_viewer:
            st.subheader("3D Valve Model")
            # Use a dummy 3D model URL
            model_url = "https://raw.githubusercontent.com/gurkan-maker/deneme/main/obje.glb"
            valve_3d_viewer(selected_valve_name, model_url)
            
        # Show simulation results if requested
        if st.session_state.show_simulation:
            st.subheader("Simulation Results")
            image_url = get_simulation_image(selected_valve_name)
            st.image(image_url, caption=f"Simulation Results for {selected_valve_name}", use_column_width=True)
    
    # Export PDF button action
    if export_btn:
        if not st.session_state.results:
            st.error("Please calculate results before exporting.")
            st.stop()
        
        try:
            # Generate plot as bytes
            fig = plot_cv_curve(
                st.session_state.results["valve"], 
                st.session_state.results["op_points"], 
                st.session_state.results["req_cvs"],
                st.session_state.results["theoretical_cvs"],
                st.session_state.results["scenario_names"]
            )
            plot_bytes = fig.to_image(format="png")
            
            # Generate PDF
            pdf_bytes = generate_pdf_report(
                st.session_state.scenarios,
                st.session_state.results["valve"],
                st.session_state.results["op_points"],
                st.session_state.results["req_cvs"],
                st.session_state.results["warnings"],
                st.session_state.results["cavitation_info"],
                plot_bytes,
                st.session_state.logo_bytes,
                st.session_state.logo_type
            )
            
            # Download button
            st.sidebar.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"valve_sizing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            st.sidebar.success("PDF report generated successfully!")
            
        except Exception as e:
            st.sidebar.error(f"PDF generation failed: {str(e)}")
            st.sidebar.text(traceback.format_exc())

if __name__ == "__main__":
    main()
