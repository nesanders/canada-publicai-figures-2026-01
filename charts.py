"""
charts.py

Main script for generating figures for the Newspaper article on AI model development.
Always cite data sources in figure captions or as annotations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Setup styles for aesthetics
# Using a clean, accessible font and color palette
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.titleweight'] = 'bold'

# Data Directory
DATA_DIR = Path(".")

def load_cost_data():
    """Load the model development cost data."""
    df = pd.read_csv(DATA_DIR / "data_cost_to_build.csv")
    # Rename unnamed first column to 'Model'
    if df.columns[0].startswith('Unnamed'):
        df.rename(columns={df.columns[0]: 'Model'}, inplace=True)
    
    # Clean cost column: remove '$' and ',' then convert to float
    df['cost_float'] = df['Cost estimate (million USD)'].replace('[\$,]', '', regex=True).astype(float)
    return df

def load_pew_data():
    """Load Pew Research datasets and merge region info."""
    concern_df = pd.read_csv(DATA_DIR / "data_pew_concern.csv")
    trust_df = pd.read_csv(DATA_DIR / "data_pew_trust.csv")
    
    # Merge Region from trust_df into concern_df if it's missing
    if 'Region' not in concern_df.columns:
        region_map = trust_df[['Country', 'Region']].drop_duplicates()
        concern_df = concern_df.merge(region_map, on='Country', how='left')
        # Fill missing regions if any (though usually they match)
        concern_df['Region'] = concern_df['Region'].fillna('Other')
        
    return concern_df, trust_df

def plot_cost_comparison(df):
    """
    Generate a bar chart comparing model development costs.
    Cites: EpochAI and author interviews.
    """
    plt.figure(figsize=(10, 8)) # Increased height for multi-line labels
    
    # Sort by cost for better visual comparison
    df_sorted = df.sort_values('cost_float', ascending=False).copy()
    
    # Create multi-line labels: Model Name + Year
    # Cleaning years to just the integer part if needed (e.g., 2025.1 -> 2025)
    df_sorted['label_with_year'] = df_sorted.apply(
        lambda x: f"{x['Model']}\n({int(float(x['Release year.month']))})", axis=1
    )
    
    # Use different colors for corporate vs public
    colors = ['#1f77b4' if m == 'Corporate' else '#ff7f0e' for m in df_sorted['Development model']]
    
    bars = plt.bar(df_sorted['label_with_year'], df_sorted['cost_float'], color=colors)
    
    plt.ylabel("Estimated Cost (Million USD)", fontsize=12)
    plt.title("The Cost Contrast: Training Frontier and PublicAI Models (2023-2025)", fontsize=16, pad=20)
    
    # Rotate x labels slightly
    plt.xticks(rotation=45, ha='center')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f"${int(yval)}M", ha='center', va='bottom', fontweight='bold')

    # Legend for operating model
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#1f77b4', lw=4, label='Corporate Model'),
                       Line2D([0], [0], color='#ff7f0e', lw=4, label='Public/Open Model')]
    plt.legend(handles=legend_elements, loc='upper right', frameon=False)

    plt.annotate("Data Source: EpochAI (2026) and Apertus project", 
                 xy=(1, -0.25), xycoords='axes fraction', ha='right', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig("figure_cost_comparison.png", dpi=300)
    print("Saved figure_cost_comparison.png")

def plot_pew_trust(df):
    """
    Generate a plot showing international trust in AI.
    Cites: Pew Research Center (2025).
    """
    plt.figure(figsize=(10, 8))
    # Sort by Region then trust
    df_sorted = df.sort_values(['Region', 'A lot / Some trust'], ascending=[True, True])
    countries = df_sorted['Country'].tolist()
    trust = df_sorted['A lot / Some trust'].tolist()
    regions = df_sorted['Region'].tolist()
    
    # Map regions to colors
    palette = {'West': '#4c72b0', 'East': '#c44e52', 'Other': '#8c8c8c'}
    colors = [palette.get(r, '#8c8c8c') if c != 'Canada' else '#d32f2f' for c, r in zip(countries, regions)]
    
    plt.barh(countries, trust, color=colors, alpha=0.8)
    
    # Set ticks and make labels clearer with region
    plt.yticks(range(len(countries)), [f"{c} ({r})" if r else c for c, r in zip(countries, regions)])
    
    # Make Canada label bold
    ax = plt.gca()
    for tick in ax.get_yticklabels():
        if 'Canada' in tick.get_text():
            tick.set_fontweight('bold')
            tick.set_color('#d32f2f')

    plt.xlabel("Share of Public with 'A lot' or 'Some' Trust in Their Government's Ability to Regulate AI (%)", fontsize=11)
    plt.title("Faith in National AI Regulation: The Global Trust Gap", fontsize=15, pad=20)
    plt.xlim(0, 1.1) 
    
    for i, v in enumerate(trust):
        plt.text(v + 0.01, i, f"{int(v*100)}%", va='center', 
                 fontweight='bold' if countries[i] == 'Canada' else 'normal')

    plt.annotate("Source: Pew Research Center (2025)", 
                 xy=(1, -0.1), xycoords='axes fraction', ha='right', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig("figure_pew_trust.png", dpi=300)
    print("Saved figure_pew_trust.png")

def plot_pew_concern(df):
    """
    Generate a plot showing concern vs excitement about AI, faceted by region.
    Cites: Pew Research Center (2025).
    """
    regions = df['Region'].unique()
    n_regions = len(regions)
    
    # Calculate height ratios based on the number of countries in each region
    region_counts = [len(df[df['Region'] == r]) for r in regions]
    
    # Scale figure height based on total countries
    total_countries = sum(region_counts)
    fig_height = max(10, total_countries * 0.4 + n_regions * 1.5)
    
    fig, axes = plt.subplots(n_regions, 1, figsize=(10, fig_height), 
                             gridspec_kw={'height_ratios': region_counts},
                             sharex=True)
    
    if n_regions == 1:
        axes = [axes]

    # Calculate x-limit based on actual data
    max_val = max(df['More Concerned'].max(), df['More Excited'].max())
    x_limit = min(1.0, max_val + 0.1)

    for ax, region in zip(axes, regions):
        region_df = df[df['Region'] == region].sort_values('More Concerned', ascending=True)
        
        countries = region_df['Country'].tolist()
        concerned = region_df['More Concerned'].tolist()
        excited = region_df['More Excited'].tolist()
        
        y_pos = np.arange(len(countries))
        height = 0.35
        
        ax.barh(y_pos - height/2, concerned, height, label='More Concerned', color='#d73027', alpha=0.7)
        ax.barh(y_pos + height/2, excited, height, label='More Excited', color='#4575b4', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(countries)
        ax.set_title(f"Region: {region}", fontsize=12, fontweight='bold', loc='left')
        
        # Highlight Canada
        for i, country in enumerate(countries):
            if country == 'Canada':
                ax.get_yticklabels()[i].set_fontweight('bold')
                ax.get_yticklabels()[i].set_color('#d32f2f')

        ax.set_xlim(0, x_limit)
        if region == regions[0]:
            ax.legend(loc='lower right', frameon=False)

    plt.xlabel("Percentage of Respondents (%)", fontsize=11)
    fig.suptitle("Global Perspectives on AI: Caution vs. Optimism", fontsize=16, fontweight='bold', y=0.98)
    
    fig.text(0.95, 0.01, "Source: Pew Research Center (2025)", 
             ha='right', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("figure_pew_concern.png", dpi=300)
    print("Saved figure_pew_concern.png")

def plot_pew_concern_gap(df):
    """
    Generate a simplified plot showing the gap (Concerned - Excited), faceted by region.
    Positive values mean more concern than excitement.
    """
    # Calculate the gap
    df['Gap'] = df['More Concerned'] - df['More Excited']
    
    regions = df['Region'].unique()
    n_regions = len(regions)
    region_counts = [len(df[df['Region'] == r]) for r in regions]
    
    total_countries = sum(region_counts)
    fig_height = max(10, total_countries * 0.4 + n_regions * 1.5)
    
    fig, axes = plt.subplots(n_regions, 1, figsize=(10, fig_height), 
                             gridspec_kw={'height_ratios': region_counts},
                             sharex=True)
    
    if n_regions == 1:
        axes = [axes]

    # Calculate symmetrical x-limit based on data
    abs_max_gap = abs(df['Gap']).max()
    x_limit = abs_max_gap + 0.1

    for ax, region in zip(axes, regions):
        region_df = df[df['Region'] == region].sort_values('Gap', ascending=True)
        
        countries = region_df['Country'].tolist()
        gap_vals = region_df['Gap'].tolist()
        
        y_pos = np.arange(len(countries))
        
        # Color based on sign
        colors = ['#d73027' if g > 0 else '#4575b4' for g in gap_vals]
        
        ax.barh(y_pos, gap_vals, color=colors, alpha=0.7)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(countries)
        ax.set_title(f"Region: {region}", fontsize=12, fontweight='bold', loc='left')
        
        # Highlight Canada
        for i, country in enumerate(countries):
            if country == 'Canada':
                ax.get_yticklabels()[i].set_fontweight('bold')
                ax.get_yticklabels()[i].set_color('#d32f2f')

        ax.set_xlim(-x_limit, x_limit)

    plt.xlabel("Net Concern (Concerned minus Excited, % points)", fontsize=11)
    fig.suptitle("Global Perspectives on AI: Net Concern by Country", fontsize=16, fontweight='bold', y=0.98)
    
    fig.text(0.95, 0.01, "Source: Pew Research Center (2025)", 
             ha='right', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("figure_pew_concern_gap.png", dpi=300)
    print("Saved figure_pew_concern_gap.png")

def plot_parameter_growth():
    """
    Plot model size growth over time for AI language models.
    Cites: EpochAI (2025) and Apertus project.
    """
    try:
        df = pd.read_csv(DATA_DIR / "data_epochai_all_ai_models.csv")
        # Clean 'Parameters' and 'Publication date'
        df['Publication date'] = pd.to_datetime(df['Publication date'], errors='coerce')
        df['Parameters'] = pd.to_numeric(df['Parameters'], errors='coerce')
        
        # Drop rows with missing crucial data
        df = df.dropna(subset=['Publication date', 'Parameters'])
        
        # Filter for AI language and multimodal models (filtering for LLM-like domains/tasks)
        target_domains = ['Language', 'Multimodal', 'Vision']
        target_tasks = ['Language modeling/generation', 'Chat', 'Conversation', 'Question answering', 
                        'Multimodal language modeling']
        
        # Simple keywords filter to match language/foundational models
        mask = (df['Domain'].str.contains('Language|Multimodal', case=False, na=False) | 
                df['Task'].str.contains('Language modeling|Chat|QA|Question answering', case=False, na=False))
        df = df[mask]

        # Filter for models from 2018 onwards
        df = df[df['Publication date'] >= '2018-01-01']
        
        # Prepare time variables for LOWESS
        start_date = df['Publication date'].min()
        df['days'] = (df['Publication date'] - start_date).dt.days
        
        plt.figure(figsize=(12, 7))
        
        # Scatter of relevant language models (semi-transparent)
        plt.scatter(df['Publication date'], df['Parameters'], alpha=0.2, color='gray', s=20, label='AI Language Models')
        
        # Highlight Frontier Models if they exist in dataset
        if 'Frontier model' in df.columns:
            frontier_mask = df['Frontier model'].fillna(False).astype(str).str.upper() == 'TRUE'
            # Intersect with our language model filter
            frontier_mask = frontier_mask & mask
            frontier = df[frontier_mask].sort_values('days')
            plt.scatter(frontier['Publication date'], frontier['Parameters'], alpha=0.6, color='#1f77b4', s=60, label='Frontier AI Models')
            
            # Trend for Frontier Models
            if len(frontier) > 5:
                lowess_frontier = sm.nonparametric.lowess(np.log10(frontier['Parameters']), frontier['days'], frac=0.4)
                plt.plot(start_date + pd.to_timedelta(lowess_frontier[:, 0], unit='D'), 
                         10**lowess_frontier[:, 1], color='#1f77b4', lw=2, label='Frontier Trend')

        # Overall trend for all filtered language models
        df_sorted = df.sort_values('days')
        lowess_all = sm.nonparametric.lowess(np.log10(df_sorted['Parameters']), df_sorted['days'], frac=0.3)
        plt.plot(start_date + pd.to_timedelta(lowess_all[:, 0], unit='D'), 
                 10**lowess_all[:, 1], "r--", alpha=0.6, label='General Trend')

        # Add Apertus manually (outlier)
        apertus_date = pd.to_datetime('2025-03-01')
        apertus_params = 40e9
        plt.scatter(apertus_date, apertus_params, color='#ff7f0e', s=150, marker='*', label='Apertus (Public Model)', zorder=5)
        plt.annotate('Apertus (40B)\nPublic/Open', 
                     xy=(apertus_date, apertus_params), 
                     xytext=(10, 10), textcoords='offset points',
                     color='#d35400', fontweight='bold', fontsize=10)

        # Formatting
        plt.yscale('log')
        plt.ylabel("Model Size (Number of Parameters)", fontsize=12)
        plt.xlabel("Publication Date", fontsize=12)
        plt.title("Scaling Up: Growth in the Size of AI Models (2018-2025)", fontsize=16, pad=20)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(loc='upper left', frameon=False)
        
        plt.annotate("Source: EpochAI (2025) and Apertus project.", 
                     xy=(1, -0.15), xycoords='axes fraction', ha='right', fontsize=9, style='italic')

        plt.tight_layout()
        plt.savefig("figure_parameter_growth.png", dpi=300)
        print("Saved figure_parameter_growth.png")
        
    except Exception as e:
        print(f"Error generating parameter growth plot: {e}")

if __name__ == "__main__":
    try:
        # Load all data
        cost_df = load_cost_data()
        concern_df, trust_df = load_pew_data()
        
        # Generate figures
        plot_cost_comparison(cost_df)
        plot_pew_trust(trust_df)
        plot_pew_concern(concern_df)
        plot_pew_concern_gap(concern_df)
        plot_parameter_growth()
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
