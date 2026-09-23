"""
src/interface.py - Rich Terminal User Interface (CLI_TUI Paradigm).
Interactive console for mirante_tecnologia_data_scientist_bridge_project.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.core_engine import create_engine

console = Console()

def run_cli():
    console.print(Panel(
        "[bold cyan]MIRANTE_TECNOLOGIA_DATA_SCIENTIST_BRIDGE_PROJECT[/bold cyan]\n"
        "[dim]Mirante Tecnologia requires an enterprise-grade Causal & Survival Lifecycle Analytics architecture u...[/dim]",
        title="GP-170 Terminal Dashboard",
        border_style="green"
    ))
    
    engine = create_engine()
    df = engine.execute_analysis()
    
    table = Table(title="Live Execution Metrics")
    for col in df.columns:
        table.add_column(col, style="cyan")
        
    for _, row in df.iterrows():
        table.add_row(*[str(val) for val in row])
        
    console.print(table)

if __name__ == "__main__":
    run_cli()