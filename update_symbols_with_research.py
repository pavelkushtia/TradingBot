#!/usr/bin/env python3
"""
Symbol Update Script using LangGraph Research
Updates trading bot symbols based on LangGraph market research
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from trading_bot.research.langgraph_trading_workflow import TradingResearchWorkflow

console = Console()


def update_env_file(symbols: list, backup: bool = True) -> bool:
    """Update the .env file with new symbols"""
    env_file = Path('.env')
    env_template = Path('config.env.template')
    
    # Create .env from template if it doesn't exist
    if not env_file.exists() and env_template.exists():
        console.print("[yellow]Creating .env from template...[/yellow]")
        with open(env_template) as template:
            with open(env_file, 'w') as env:
                env.write(template.read())
    
    if not env_file.exists():
        console.print("[red]No .env file found and no template available[/red]")
        return False
    
    # Backup existing file
    if backup and env_file.exists():
        backup_file = Path(f'.env.backup.{int(asyncio.get_event_loop().time())}')
        with open(env_file) as original:
            with open(backup_file, 'w') as backup_f:
                backup_f.write(original.read())
        console.print(f"[green]Backed up .env to {backup_file}[/green]")
    
    # Read current env file
    lines = []
    symbols_updated = False
    
    with open(env_file) as f:
        lines = f.readlines()
    
    # Update SYMBOLS line
    symbols_str = ','.join(symbols)
    
    for i, line in enumerate(lines):
        if line.startswith('SYMBOLS='):
            lines[i] = f'SYMBOLS={symbols_str}\n'
            symbols_updated = True
            break
    
    # If SYMBOLS line doesn't exist, add it
    if not symbols_updated:
        lines.append(f'SYMBOLS={symbols_str}\n')
    
    # Write updated file
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    console.print(f"[green]✅ Updated .env with symbols: {symbols_str}[/green]")
    return True


def save_research_report(result: dict, filename: Optional[str] = None) -> None:
    """Save detailed research report"""
    if not filename:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'research_report_{timestamp}.json'
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'success': result['success'],
        'final_symbols': result['final_symbols'],
        'confidence_scores': result['confidence_scores'],
        'current_positions': result['current_positions'],
        'market_research': result['market_research'],
        'analyzed_symbols': result['analyzed_symbols'],
        'error': result.get('error', '')
    }
    
    with open(filename, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    console.print(f"[green]📄 Research report saved to {filename}[/green]")


async def main() -> None:
    """Main function to run symbol research and update"""
    
    console.print(Panel.fit(
        "[bold blue]🚀 LangGraph Trading Symbol Research[/bold blue]\n"
        "Integrating AI-powered market research with your trading bot",
        title="🤖 AI Trading Research"
    ))
    
    try:
        # Initialize the research workflow
        console.print("\n[cyan]Initializing LangGraph research workflow...[/cyan]")
        workflow = TradingResearchWorkflow()
        
        # Execute research
        console.print("[cyan]Executing AI-powered market research...[/cyan]")
        result = await workflow.execute_research(
            "Analyze current market trends and recommend the best trading symbols for my portfolio"
        )
        
        if result['success']:
            symbols = result['final_symbols']
            
            console.print(f"\n[bold green]🎯 Research completed successfully![/bold green]")
            console.print(f"[green]Recommended symbols ({len(symbols)}): {', '.join(symbols)}[/green]")
            
            # Ask user if they want to update
            console.print(f"\n[green]✅ Symbol selection completed![/green]")
            console.print(f"[cyan]Selected {len(symbols)} symbols total[/cyan]")
            console.print(f"[cyan]Symbols: {', '.join(symbols)}[/cyan]")
            
            response = input(f"\nUpdate .env file with your {len(symbols)} selected symbols? (y/N): ")
            
            if response.lower() == 'y':
                # Update .env file
                if update_env_file(symbols):
                    console.print("\n[bold green]✅ Trading bot symbols updated successfully![/bold green]")
                    console.print("[cyan]Restart the trading bot to use new symbols:[/cyan]")
                    console.print("[cyan]  ./start.sh run[/cyan]")
                else:
                    console.print("[red]❌ Failed to update symbols[/red]")
            else:
                console.print("[yellow]Symbols not updated.[/yellow]")
            
            # Save research report
            save_research_report(result)
            
            # Display summary
            console.print(Panel.fit(
                f"[bold green]Research Summary[/bold green]\n"
                f"• Current positions: {len(result['current_positions'])}\n"
                f"• Recommended symbols: {len(symbols)}\n"
                f"• Market research: {'Completed' if result['market_research'] else 'Failed'}\n"
                f"• Analysis confidence: {'High' if any(c > 0.7 for c in result['confidence_scores'].values()) else 'Medium'}\n"
                f"• Error status: {result.get('error', 'None')}",
                title="📊 Results"
            ))
            
        else:
            console.print(f"[red]❌ Research failed: {result['error']}[/red]")
            console.print("[yellow]You can still manually update symbols in .env file[/yellow]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Research cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        console.print("[yellow]Please check your LangGraph cluster is running[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
