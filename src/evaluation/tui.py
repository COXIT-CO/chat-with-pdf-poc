from rich.console import Console
from rich.panel import Panel


def prompt_user_for_evaluation_details() -> str:
    """
    Prompt the user to enter details about the current evaluation run using rich formatting.

    Returns:
        str: User-provided description of the evaluation experiment
    """
    console = Console()

    console.print(
        Panel(
            "[bold]Please describe what's different or special about this evaluation run.[/bold]\n"
            "For example: changes in model parameters, different data sources, etc.\n"
            "This will be saved with the evaluation results for future reference.",
            title="EVALUATION RUN DETAILS",
            expand=False
        )
    )

    details = ""
    while not details.strip():
        details = console.input("[bold cyan]Evaluation details:[/bold cyan] ").strip()
        if not details:
            console.print("[yellow]Please provide some details to identify this evaluation run.[/yellow]")

    return details
