"""
src/interface.py - Rich Terminal User Interface (CLI_TUI Paradigm).
Interactive console for Mirante Lakehouse SLA Survival Engine.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def run_cli():
    console.print(Panel(
        "[bold cyan]MIRANTE LAKEHOUSE SLA SURVIVAL ENGINE[/bold cyan]\n"
        "[dim]Causal & Survival Lifecycle Analytics for Spark & Delta Lake Pipelines[/dim]",
        title="Production Telemetry & FinOps Dashboard",
        border_style="green"
    ))

if __name__ == "__main__":
    run_cli()