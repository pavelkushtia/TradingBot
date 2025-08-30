#!/usr/bin/env python3
"""
LangGraph Trading Research Workflow Integration
Integrates with existing LangGraph setup for intelligent trading symbol research
"""

import asyncio
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict
from decimal import Decimal

import requests  # type: ignore
from langgraph.graph import StateGraph, END  # type: ignore
from rich.console import Console
from rich.table import Table
import numpy as np  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

# Import trading bot components
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

console = Console()


class TradingResearchState(TypedDict):
    """State for the trading research workflow"""
    query: str
    current_positions: List[Dict[str, Any]]
    market_research: str
    trending_symbols: List[str]
    analyzed_symbols: List[Dict[str, Any]]
    final_recommendations: List[str]
    confidence_scores: Dict[str, float]
    step: str
    error: str


class TradingResearchWorkflow:
    """LangGraph workflow for intelligent trading symbol research"""
    
    def __init__(self, alpaca_config: Optional[Dict] = None):
        """Initialize with Alpaca configuration"""
        self.alpaca_config = alpaca_config or self._load_alpaca_config()
        self.langgraph_endpoints = {
            'llm_jetson': 'http://192.168.1.177:11434/api/generate',
            'llm_cpu': 'http://192.168.1.81:11435/api/generate', 
            'web_search': 'http://192.168.1.190:8082/web_search',
            'web_scrape': 'http://192.168.1.190:8082/web_scrape',
            'embeddings': 'http://192.168.1.81:9002/embeddings'
        }
        self.workflow = self._create_workflow()
    
    def _load_alpaca_config(self) -> Dict:
        """Load Alpaca configuration from environment"""
        from dotenv import load_dotenv
        load_dotenv()
        
        return {
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
            'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        }
    
    async def _get_alpaca_positions(self) -> List[Dict[str, Any]]:
        """Fetch current positions from Alpaca API"""
        if not self.alpaca_config['api_key'] or not self.alpaca_config['secret_key']:
            console.print("[yellow]Warning: No Alpaca credentials found, using mock data[/yellow]")
            return self._get_mock_positions()
        
        headers = {
            'APCA-API-KEY-ID': self.alpaca_config['api_key'],
            'APCA-API-SECRET-KEY': self.alpaca_config['secret_key'],
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.alpaca_config['base_url']}/v2/positions",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        positions = await response.json()
                        
                        # Sort by position value (descending) and take first 14
                        sorted_positions = sorted(
                            positions,
                            key=lambda x: abs(float(x.get('market_value', 0))),
                            reverse=True
                        )[:14]
                        
                        return [
                            {
                                'symbol': pos.get('symbol'),
                                'side': pos.get('side'),
                                'quantity': float(pos.get('qty', 0)),
                                'market_value': float(pos.get('market_value', 0)),
                                'unrealized_pl': float(pos.get('unrealized_pl', 0)),
                                'avg_entry_price': float(pos.get('avg_entry_price', 0)),
                                'percent_of_portfolio': float(pos.get('market_value', 0)) / 100000 * 100  # Assuming 100k portfolio
                            }
                            for pos in sorted_positions
                        ]
                    else:
                        console.print(f"[red]Failed to fetch positions: {response.status}[/red]")
                        return self._get_mock_positions()
        except Exception as e:
            console.print(f"[red]Error fetching positions: {e}[/red]")
            return self._get_mock_positions()
    
    def _get_mock_positions(self) -> List[Dict[str, Any]]:
        """Return mock positions for testing"""
        return [
            {
                'symbol': 'AAPL',
                'side': 'long',
                'quantity': 50,
                'market_value': 9500.0,
                'unrealized_pl': 150.0,
                'avg_entry_price': 190.0,
                'percent_of_portfolio': 9.5
            },
            {
                'symbol': 'GOOGL', 
                'side': 'long',
                'quantity': 30,
                'market_value': 8200.0,
                'unrealized_pl': -200.0,
                'avg_entry_price': 275.0,
                'percent_of_portfolio': 8.2
            },
            {
                'symbol': 'MSFT',
                'side': 'long', 
                'quantity': 25,
                'market_value': 7800.0,
                'unrealized_pl': 300.0,
                'avg_entry_price': 310.0,
                'percent_of_portfolio': 7.8
            }
        ]
    
    async def _call_llm(self, prompt: str, model: str = 'jetson') -> str:
        """Call LLM service with better error handling"""
        endpoint = self.langgraph_endpoints['llm_jetson'] if model == 'jetson' else self.langgraph_endpoints['llm_cpu']
        model_name = 'llama3.2:3b' if model == 'jetson' else 'mistral:7b'
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json={'model': model_name, 'prompt': prompt, 'stream': False},
                    timeout=120  # Longer timeout for CPU model
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')
                        if response_text and len(response_text.strip()) > 10:
                            return str(response_text)
                        else:
                            console.print(f"[yellow]⚠️ {model} returned empty/short response[/yellow]")
                            return "EMPTY_RESPONSE"
                    else:
                        console.print(f"[red]❌ {model} service error: {response.status}[/red]")
                        return "SERVICE_ERROR"
        except Exception as e:
            console.print(f"[red]❌ {model} connection failed: {str(e)}[/red]")
            return "CONNECTION_ERROR"
    
    async def _web_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform web search using LangGraph tools server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.langgraph_endpoints['web_search'],
                    json={'query': query, 'max_results': max_results},
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return dict(result)
                    else:
                        return {'error': f'Search failed with status {response.status}'}
        except Exception as e:
            return {'error': f'Search error: {str(e)}'}
    
    async def _validate_symbols_semantically(self, symbols: List[str]) -> List[str]:
        """Use embeddings server to validate symbols are actually stock tickers"""
        try:
            # Create validation prompts
            validation_texts = [f"{symbol} stock ticker symbol" for symbol in symbols]
            reference_texts = ["AAPL Apple stock", "GOOGL Google stock", "TSLA Tesla stock", "MSFT Microsoft stock"]
            
            all_texts = validation_texts + reference_texts
            
            # Get embeddings from your cluster
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.langgraph_endpoints['embeddings'],
                    json={'texts': all_texts, 'model': 'default'},
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embeddings = data['embeddings']
                        
                        # Calculate similarity to known stock patterns
                        import numpy as np
                        from sklearn.metrics.pairwise import cosine_similarity
                        
                        symbol_embeddings = np.array(embeddings[:len(symbols)])
                        reference_embeddings = np.array(embeddings[len(symbols):])
                        
                        # Calculate similarity to reference stock embeddings
                        similarities = cosine_similarity(symbol_embeddings, reference_embeddings)
                        avg_similarities = similarities.mean(axis=1)
                        
                        # Filter symbols with good similarity scores (>0.3)
                        validated = []
                        for i, symbol in enumerate(symbols):
                            if avg_similarities[i] > 0.3:  # Semantic similarity threshold
                                validated.append(symbol)
                        
                        console.print(f"[blue]🧠 Semantic validation: {len(validated)}/{len(symbols)} symbols validated[/blue]")
                        return validated
                    else:
                        console.print("[yellow]⚠️ Embeddings server unavailable, using basic validation[/yellow]")
                        return self._basic_symbol_validation(symbols)
                        
        except Exception as e:
            console.print(f"[yellow]⚠️ Semantic validation failed: {e}[/yellow]")
            return self._basic_symbol_validation(symbols)
    
    def _basic_symbol_validation(self, symbols: List[str]) -> List[str]:
        """Basic validation without semantic analysis"""
        # Comprehensive garbage list
        garbage_symbols = {
            'I', 'JSON', 'NULL', 'TRUE', 'FALSE', 'OR', 'AND', 'THE', 'FOR', 'TO', 'API', 'HTTP', 'URL',
            'LLM', 'AI', 'ML', 'CPU', 'GPU', 'RAM', 'SSD', 'HDD', 'USB', 'WIFI', 'HTML', 'CSS', 'JS',
            'ERROR', 'FAIL', 'EMPTY', 'SERVICE', 'CONNECTION', 'SEARCH', 'RESULTS', 'DATA', 'INFO',
            'TITLE', 'LINK', 'PAGE', 'SITE', 'WEB', 'NET', 'COM', 'ORG', 'GOV', 'EDU', 'IO', 'CO'
        }
        
        valid_symbols = []
        for symbol in symbols:
            # Filter out obviously wrong symbols
            if (len(symbol) >= 2 and len(symbol) <= 5 and 
                symbol.isalpha() and symbol.isupper() and 
                symbol not in garbage_symbols):
                valid_symbols.append(symbol)
        return valid_symbols
    
    async def _research_top_performers(self) -> List[str]:
        """Research current top performing stocks as intelligent fallback"""
        try:
            console.print("[blue]🔍 Researching current market leaders...[/blue]")
            
            # Comprehensive search for market leaders to get 25+ symbols
            search_queries = [
                "S&P 500 top performers today",
                "NASDAQ best stocks this month", 
                "market leaders by volume",
                "most active stocks today",
                "biggest gainers stock market",
                "trending stocks high volume",
                "institutional favorites stocks",
                "analyst upgraded stocks",
                "momentum stocks breaking out",
                "blue chip stocks performing well"
            ]
            
            all_symbols = set()
            for query in search_queries:
                result = await self._web_search(query, max_results=3)
                if 'results' in result:
                    for item in result['results']:
                        text = f"{item.get('title', '')} {item.get('snippet', '')}"
                        # Extract symbols from search results
                        import re
                        found_symbols = re.findall(r'\b([A-Z]{2,5})\b', text)
                        all_symbols.update(found_symbols)
            
            # Validate the found symbols
            validated = self._basic_symbol_validation(list(all_symbols))
            
            # Return up to 25 symbols for good selection
            if validated:
                console.print(f"[green]✅ Found {len(validated)} market leaders from research[/green]")
                return validated[:25]  # Return up to 25 for selection
            else:
                # Absolute last resort - expand to more current market leaders
                console.print("[yellow]Using expanded market leaders as final fallback[/yellow]")
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'LLY', 'V', 
                       'JNJ', 'WMT', 'JPM', 'XOM', 'UNH', 'MA', 'PG', 'HD', 'CVX', 'ABBV',
                       'BAC', 'AVGO', 'KO', 'PEP', 'COST']  # Top 25 by market cap
                
        except Exception as e:
            console.print(f"[red]❌ Research fallback failed: {e}[/red]")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    def _create_workflow(self) -> StateGraph:  # type: ignore
        """Create the LangGraph trading research workflow"""
        
        async def fetch_positions_node(state: TradingResearchState) -> TradingResearchState:
            """Fetch current Alpaca positions (first 14 by value)"""
            try:
                state['step'] = 'Fetching current portfolio positions...'
                positions = await self._get_alpaca_positions()
                state['current_positions'] = positions
                
                console.print(f"[green]✅ Fetched {len(positions)} positions[/green]")
                
                # Display positions
                if positions:
                    table = Table(title="Current Portfolio Positions")
                    table.add_column("Symbol", style="cyan")
                    table.add_column("Quantity", style="magenta")
                    table.add_column("Market Value", style="green")
                    table.add_column("P&L", style="yellow")
                    table.add_column("% Portfolio", style="blue")
                    
                    for pos in positions[:5]:  # Show top 5
                        pnl_color = "green" if pos['unrealized_pl'] >= 0 else "red"
                        table.add_row(
                            pos['symbol'],
                            str(int(pos['quantity'])),
                            f"${pos['market_value']:,.2f}",
                            f"[{pnl_color}]${pos['unrealized_pl']:,.2f}[/{pnl_color}]",
                            f"{pos['percent_of_portfolio']:.1f}%"
                        )
                    
                    console.print(table)
                
            except Exception as e:
                state['error'] = f"Failed to fetch positions: {str(e)}"
                console.print(f"[red]❌ {state['error']}[/red]")
            
            return state
        
        async def market_research_node(state: TradingResearchState) -> TradingResearchState:
            """Research current market trends and hot stocks"""
            try:
                state['step'] = 'Researching market trends and hot stocks...'
                
                # Search for SHORT-TERM trading opportunities
                current_date = datetime.now()
                search_queries = [
                    "day trading stocks with high volume today",
                    "swing trading opportunities this week",
                    "momentum stocks breaking out", 
                    "stocks with high volatility for trading",
                    "most active stocks for day traders",
                    "short term trading stocks trending",
                    "stocks moving on news today",
                    "volatile stocks good for trading"
                ]
                
                all_search_results = []
                for query in search_queries[:4]:  # More searches for short-term opportunities
                    console.print(f"🔍 Searching: {query}")
                    result = await self._web_search(query, max_results=5)
                    if 'results' in result:
                        all_search_results.extend(result['results'][:3])  # Top 3 per query
                
                # Format search results for LLM analysis
                search_summary = "\n".join([
                    f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}\n"
                    for r in all_search_results[:8]  # Max 8 results
                ])
                
                # Analyze with LLM
                analysis_prompt = f"""
Current Date: {datetime.now().strftime('%Y-%m-%d')}

Market Research Data:
{search_summary}

Based on this market research, identify:
1. Top 5 trending/hot stock symbols currently in the market
2. Key market trends driving these stocks
3. Sectors showing strength

Please provide a structured analysis with:
- List of 5 hot stock symbols (just the ticker symbols)
- Brief explanation of why each is trending
- Overall market sentiment

Format your response clearly with stock symbols in a list.
"""
                
                analysis = await self._call_llm(analysis_prompt, model='cpu')  # Use powerful model
                state['market_research'] = analysis
                
                console.print("[green]✅ Market research completed[/green]")
                
            except Exception as e:
                state['error'] = f"Market research failed: {str(e)}"
                state['market_research'] = "Unable to complete market research"
                console.print(f"[red]❌ {state['error']}[/red]")
            
            return state
        
        async def extract_symbols_node(state: TradingResearchState) -> TradingResearchState:
            """Extract trending symbols using advanced LLM techniques and semantic validation"""
            try:
                state['step'] = 'Extracting trending symbols with AI validation...'
                
                # Advanced prompt engineering with clear structure and examples
                current_date = datetime.now().strftime('%B %Y')
                extraction_prompt = f"""You are a financial analyst extracting stock ticker symbols from market research.

TASK: Extract exactly 5 trending stock ticker symbols from the research below.

RESEARCH DATA:
{state['market_research']}

INSTRUCTIONS:
1. Look for company names or stock symbols mentioned as "trending", "hot", "performing well", "gaining", "bullish"
2. Convert company names to their stock ticker symbols (e.g., "Apple" → "AAPL", "Tesla" → "TSLA")
3. Only include valid NYSE/NASDAQ symbols (2-5 uppercase letters)
4. Prioritize large-cap stocks over penny stocks
5. Current date context: {current_date}

OUTPUT FORMAT (return exactly this structure):
SYMBOL_1: [Symbol]
SYMBOL_2: [Symbol]  
SYMBOL_3: [Symbol]
SYMBOL_4: [Symbol]
SYMBOL_5: [Symbol]

EXAMPLE OUTPUT:
SYMBOL_1: AAPL
SYMBOL_2: MSFT
SYMBOL_3: GOOGL
SYMBOL_4: TSLA
SYMBOL_5: NVDA

YOUR OUTPUT:"""
                
                symbols_response = await self._call_llm(extraction_prompt, model='cpu')  # Use more powerful model
                
                # DEBUG: Show what LLM actually returned
                console.print(f"[cyan]🔍 LLM Response Debug:[/cyan]")
                console.print(f"[dim]{symbols_response[:200]}...[/dim]")
                
                # Parse the structured response
                import re
                symbols = []
                
                # Look for SYMBOL_N: TICKER pattern
                symbol_matches = re.findall(r'SYMBOL_\d+:\s*([A-Z]{2,5})', symbols_response)
                symbols.extend(symbol_matches)
                console.print(f"[cyan]Pattern 1 (SYMBOL_N:): {symbol_matches}[/cyan]")
                
                # If structured parsing fails, try alternative patterns
                if not symbols:
                    # Look for listed symbols with numbers
                    alt_matches = re.findall(r'\d+\.\s*([A-Z]{2,5})', symbols_response)
                    symbols.extend(alt_matches)
                    console.print(f"[cyan]Pattern 2 (numbered list): {alt_matches}[/cyan]")
                
                # If still no luck, extract any valid symbols
                if not symbols:
                    all_matches = re.findall(r'\b([A-Z]{2,5})\b', symbols_response)
                    symbols.extend(all_matches)
                    console.print(f"[cyan]Pattern 3 (any caps): {all_matches}[/cyan]")
                
                # Handle error responses first
                if symbols_response in ["EMPTY_RESPONSE", "SERVICE_ERROR", "CONNECTION_ERROR"]:
                    console.print("[yellow]⚠️ LLM service issues, using research-based approach...[/yellow]")
                    research_symbols = await self._research_top_performers()
                    state['trending_symbols'] = research_symbols[:25]  # Use first 25 for selection
                elif symbols:
                    # Validate symbols using semantic similarity (leverage your embeddings server!)
                    validated_symbols = await self._validate_symbols_semantically(symbols[:20])  # Check top 20
                    if len(validated_symbols) >= 15:
                        state['trending_symbols'] = validated_symbols[:25]  # Take 25 for selection
                    else:
                        # Mix extracted with research
                        research_symbols = await self._research_top_performers()
                        combined = list(dict.fromkeys(validated_symbols + research_symbols))  # Remove dups, preserve order
                        state['trending_symbols'] = combined[:25]
                else:
                    # If extraction completely fails, do intelligent fallback research
                    console.print("[yellow]⚠️ Symbol extraction failed, researching top performers...[/yellow]")
                    fallback_symbols = await self._research_top_performers()
                    state['trending_symbols'] = fallback_symbols[:25]
                
                console.print(f"[green]✅ Extracted & validated symbols: {', '.join(state['trending_symbols'])}[/green]")
                
            except Exception as e:
                state['error'] = f"Symbol extraction failed: {str(e)}"
                # Last resort: research current market leaders
                console.print(f"[red]❌ Error in extraction: {e}[/red]")
                state['trending_symbols'] = await self._research_top_performers()
                console.print(f"[yellow]Using researched top performers as fallback[/yellow]")
            
            return state
        
        async def analyze_symbols_node(state: TradingResearchState) -> TradingResearchState:
            """Analyze each trending symbol for trading potential"""
            try:
                state['step'] = 'Analyzing trending symbols...'
                
                current_symbols = [pos['symbol'] for pos in state['current_positions']]
                analyzed_symbols = []
                
                for symbol in state['trending_symbols']:
                    console.print(f"📊 Analyzing {symbol}...")
                    
                    # Search for specific symbol analysis (no hardcoded year!)
                    symbol_search = await self._web_search(f"{symbol} stock analysis price target", max_results=3)
                    
                    symbol_info = ""
                    if 'results' in symbol_search:
                        symbol_info = "\n".join([
                            f"{r.get('title', '')}: {r.get('snippet', '')}"
                            for r in symbol_search['results'][:2]
                        ])
                    
                    # Analyze with LLM for SHORT-TERM TRADING
                    analysis_prompt = f"""
Symbol: {symbol}
Current Portfolio Holdings: {', '.join(current_symbols)}

Recent Information:
{symbol_info}

Analyze this symbol for SHORT-TERM TRADING (days to weeks) considering:
1. Is it already in the current portfolio?
2. Recent momentum and volatility (good for trading)
3. Volume and liquidity (can we get in/out easily?)
4. News catalysts or technical breakouts
5. Short-term trading opportunity (BUY/HOLD/AVOID)

Focus on TRADING OPPORTUNITIES, not long-term investing. Look for:
- High volume and volatility 
- Recent price movements
- Technical momentum
- News-driven moves
- Liquidity for quick entries/exits

Recommendation (BUY/HOLD/AVOID): Be more aggressive with BUY recommendations for tradeable stocks.
"""
                    
                    analysis = await self._call_llm(analysis_prompt, model='jetson')
                    
                    # Determine recommendation and confidence - MORE AGGRESSIVE for trading
                    recommendation = 'BUY'  # Default to BUY for trading opportunities
                    confidence = 0.7  # Higher default confidence
                    
                    analysis_upper = analysis.upper()
                    
                    # Look for strong trading signals
                    if any(word in analysis_upper for word in ['STRONG BUY', 'EXCELLENT', 'BREAKOUT', 'MOMENTUM', 'VOLATILE', 'ACTIVE']):
                        recommendation = 'BUY'
                        confidence = 0.9
                    elif any(word in analysis_upper for word in ['BUY', 'TRADING', 'OPPORTUNITY', 'VOLUME', 'MOVE']):
                        recommendation = 'BUY'
                        confidence = 0.8
                    elif any(word in analysis_upper for word in ['AVOID', 'SELL', 'RISKY', 'DECLINE', 'FALLING']):
                        recommendation = 'AVOID'
                        confidence = 0.4
                    elif 'HOLD' in analysis_upper:
                        recommendation = 'HOLD'
                        confidence = 0.6
                    
                    analyzed_symbols.append({
                        'symbol': symbol,
                        'analysis': analysis,
                        'recommendation': recommendation,
                        'confidence': confidence,
                        'already_held': symbol in current_symbols
                    })
                
                state['analyzed_symbols'] = analyzed_symbols
                console.print(f"[green]✅ Analyzed {len(analyzed_symbols)} symbols[/green]")
                
            except Exception as e:
                state['error'] = f"Symbol analysis failed: {str(e)}"
                console.print(f"[red]❌ {state['error']}[/red]")
            
            return state
        
        async def finalize_recommendations_node(state: TradingResearchState) -> TradingResearchState:
            """Finalize trading recommendations"""
            try:
                state['step'] = 'Finalizing recommendations...'
                
                # Present ALL analyzed symbols for individual selection
                console.print(f"\n[bold blue]📋 Symbol Selection ({len(state['analyzed_symbols'])} candidates)[/bold blue]")
                console.print("[cyan]Choose which symbols to include in your trading list (y/n for each):[/cyan]\n")
                
                selected_symbols = []
                current_symbols = [pos['symbol'] for pos in state['current_positions']]
                
                # Always include current positions first
                for pos in state['current_positions']:
                    selected_symbols.append(pos['symbol'])
                    console.print(f"[green]✅ {pos['symbol']} (Current holding - auto-included)[/green]")
                
                # Present each new symbol for individual choice
                for sym_data in state['analyzed_symbols']:
                    if sym_data['symbol'] not in current_symbols:
                        symbol = sym_data['symbol']
                        rec = sym_data['recommendation']
                        conf = sym_data['confidence']
                        
                        # Show recommendation with color coding
                        rec_color = "green" if rec == "BUY" else "yellow" if rec == "HOLD" else "red"
                        
                        console.print(f"\n[bold]{symbol}[/bold] - [{rec_color}]{rec}[/{rec_color}] (Confidence: {conf:.1%})")
                        console.print(f"[dim]{sym_data['analysis'][:150]}...[/dim]")
                        
                        choice = input(f"Include {symbol} in trading list? (y/n): ").lower().strip()
                        
                        if choice == 'y':
                            selected_symbols.append(symbol)
                            console.print(f"[green]✅ Added {symbol}[/green]")
                        else:
                            console.print(f"[red]❌ Skipped {symbol}[/red]")
                        
                        # Stop if we reach 15 total (including current positions)
                        if len(selected_symbols) >= 15:
                            console.print(f"[yellow]Reached 15 symbol limit. Stopping selection.[/yellow]")
                            break
                
                state['final_recommendations'] = selected_symbols
                
                # Create confidence scores
                confidence_scores = {}
                for sym in state['analyzed_symbols']:
                    confidence_scores[sym['symbol']] = sym['confidence']
                
                state['confidence_scores'] = confidence_scores
                
                # Display final recommendations
                table = Table(title="Final Trading Symbol Recommendations")
                table.add_column("Symbol", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Recommendation", style="yellow")
                table.add_column("Confidence", style="magenta")
                
                for symbol in state['final_recommendations']:
                    status = "Current Holding" if symbol in current_symbols else "New Addition"
                    
                    # Find recommendation
                    rec_info = next((s for s in state['analyzed_symbols'] if s['symbol'] == symbol), None)
                    if rec_info:
                        recommendation = rec_info['recommendation']
                        confidence = f"{rec_info['confidence']:.1%}"
                    else:
                        recommendation = "HOLD"
                        confidence = "N/A"
                    
                    table.add_row(symbol, status, recommendation, confidence)
                
                console.print(table)
                console.print(f"\n[bold green]🎯 Final symbol list ({len(state['final_recommendations'])} symbols):[/bold green]")
                console.print(f"[cyan]{', '.join(state['final_recommendations'])}[/cyan]")
                
            except Exception as e:
                state['error'] = f"Finalization failed: {str(e)}"
                state['final_recommendations'] = [pos['symbol'] for pos in state['current_positions']]  # Fallback
                console.print(f"[red]❌ {state['error']}[/red]")
            
            return state
        
        # Build the workflow graph
        from langgraph.graph import StateGraph  # Import here to avoid conflicts
        workflow = StateGraph(TradingResearchState)
        
        # Add nodes
        workflow.add_node("fetch_positions", fetch_positions_node)
        workflow.add_node("research_market", market_research_node)
        workflow.add_node("extract_symbols", extract_symbols_node)
        workflow.add_node("analyze_symbols", analyze_symbols_node)
        workflow.add_node("finalize_recs", finalize_recommendations_node)
        
        # Set flow
        workflow.set_entry_point("fetch_positions")
        workflow.add_edge("fetch_positions", "research_market")
        workflow.add_edge("research_market", "extract_symbols")
        workflow.add_edge("extract_symbols", "analyze_symbols")
        workflow.add_edge("analyze_symbols", "finalize_recs")
        workflow.add_edge("finalize_recs", END)
        
        return workflow.compile()
    
    async def execute_research(self, query: str = "Find hot trading stocks for my portfolio") -> Dict[str, Any]:
        """Execute the complete trading research workflow"""
        
        console.print(f"\n[bold blue]🚀 Starting Trading Research Workflow[/bold blue]")
        console.print(f"[cyan]Query: {query}[/cyan]\n")
        
        initial_state = {
            'query': query,
            'current_positions': [],
            'market_research': '',
            'trending_symbols': [],
            'analyzed_symbols': [],
            'final_recommendations': [],
            'confidence_scores': {},
            'step': 'Starting...',
            'error': ''
        }
        
        try:
            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)
            
            console.print(f"\n[bold green]✅ Trading Research Workflow Completed![/bold green]")
            
            return {
                'success': True,
                'final_symbols': result['final_recommendations'],
                'confidence_scores': result['confidence_scores'],
                'current_positions': result['current_positions'],
                'market_research': result['market_research'],
                'analyzed_symbols': result['analyzed_symbols'],
                'error': result.get('error', '')
            }
            
        except Exception as e:
            console.print(f"[red]❌ Workflow failed: {str(e)}[/red]")
            return {
                'success': False,
                'error': str(e),
                'final_symbols': [],
                'confidence_scores': {},
                'current_positions': [],
                'market_research': '',
                'analyzed_symbols': []
            }


async def main():
    """Test the trading research workflow"""
    
    # Initialize workflow
    workflow = TradingResearchWorkflow()
    
    # Execute research
    result = await workflow.execute_research(
        "Research hot stocks and update my trading symbols based on current market trends"
    )
    
    if result['success']:
        print(f"\n🎯 Recommended symbols: {', '.join(result['final_symbols'])}")
        if result['error']:
            print(f"⚠️  Warning: {result['error']}")
    else:
        print(f"❌ Failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
