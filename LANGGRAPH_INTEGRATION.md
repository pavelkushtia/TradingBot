# 🤖 LangGraph AI Integration for Trading Bot

This document describes the integration between your **Trading Bot** and **LangGraph AI Cluster** for intelligent symbol selection and market research.

## 🚀 Overview

The integration adds AI-powered market research capabilities to your trading bot by leveraging your existing LangGraph infrastructure for:

1. **Smart Position Management**: Automatically fetch your current Alpaca positions (top 14 by value)
2. **Market Intelligence**: Use web search and scraping to find trending stocks
3. **AI Analysis**: LLM-powered analysis of market trends and stock opportunities  
4. **Intelligent Recommendations**: Generate optimized symbol lists for trading
5. **Seamless Integration**: Automatically update your trading bot configuration

## 🏗️ Architecture

```
Trading Bot <--> LangGraph Workflow <--> LangGraph Cluster
     ↓                    ↓                      ↓
[Alpaca API]     [Symbol Research]     [Jetson LLM + CPU LLM]
     ↓                    ↓                      ↓  
[Positions]      [Market Analysis]     [Tools Server + Embeddings]
     ↓                    ↓                      ↓
[.env Update]    [Recommendations]     [Web Search + Scraping]
```

## 🔧 Setup & Requirements

### Prerequisites

1. **LangGraph Cluster Running**: Your local AI infrastructure must be operational
2. **Alpaca Account**: Paper trading account with API credentials
3. **Network Access**: Trading bot can reach LangGraph endpoints

### Dependencies

The integration automatically installs required packages:
- `langgraph>=0.1.0`
- `langchain>=0.1.0` 
- `aiohttp>=3.8.0` (already included)

### LangGraph Endpoints Used

- **Jetson LLM**: `http://192.168.1.177:11434/api/generate` (fast inference)
- **CPU LLM**: `http://192.168.1.81:11435/api/generate` (heavy analysis)
- **Web Search**: `http://192.168.1.190:8082/web_search`
- **Web Scraping**: `http://192.168.1.190:8082/web_scrape`
- **Embeddings**: `http://192.168.1.81:9002/embeddings`

## 🎯 Usage

### Quick Start

```bash
# Run AI-powered symbol research
./start.sh research

# Traditional trading (with manually selected symbols)
./start.sh run

# Web dashboard
./start.sh web
```

### Research Workflow Process

1. **Fetch Positions**: Gets your current Alpaca positions (top 14 by value)
2. **Market Research**: Searches for trending stocks and market news
3. **Extract Symbols**: Uses LLM to identify hot stock symbols
4. **Analyze Symbols**: Individual analysis of each trending symbol
5. **Generate Recommendations**: Creates optimized symbol list
6. **Update Configuration**: Updates `.env` file with new symbols

### Example Output

```
🚀 Starting Trading Research Workflow
Query: Analyze current market trends and recommend the best trading symbols

✅ Fetched 3 positions

┌─────────┬──────────┬──────────────┬──────────┬─────────────┐
│ Symbol  │ Quantity │ Market Value │   P&L    │ % Portfolio │
├─────────┼──────────┼──────────────┼──────────┼─────────────┤
│ AAPL    │ 50       │ $9,500.00    │ $150.00  │ 9.5%        │
│ GOOGL   │ 30       │ $8,200.00    │ -$200.00 │ 8.2%        │
│ MSFT    │ 25       │ $7,800.00    │ $300.00  │ 7.8%        │
└─────────┴──────────┴──────────────┴──────────┴─────────────┘

🔍 Searching: trending stocks today 2024
🔍 Searching: hot stocks to buy now
✅ Market research completed
✅ Extracted symbols: NVDA, TSLA, AMZN, META, GOOGL
📊 Analyzing NVDA...
📊 Analyzing TSLA...
📊 Analyzing AMZN...
📊 Analyzing META...
📊 Analyzing GOOGL...
✅ Analyzed 5 symbols

┌────────┬────────────────┬──────────────┬────────────┐
│ Symbol │     Status     │ Recommendation │ Confidence │
├────────┼────────────────┼──────────────┼────────────┤
│ AAPL   │ Current Holding│ HOLD         │ N/A        │
│ GOOGL  │ Current Holding│ HOLD         │ 80.0%      │
│ MSFT   │ Current Holding│ HOLD         │ N/A        │
│ NVDA   │ New Addition   │ BUY          │ 80.0%      │
│ TSLA   │ New Addition   │ BUY          │ 80.0%      │
│ AMZN   │ New Addition   │ BUY          │ 80.0%      │
│ META   │ New Addition   │ BUY          │ 80.0%      │
└────────┴────────────────┴──────────────┴────────────┘

🎯 Final symbol list (7 symbols):
AAPL, GOOGL, MSFT, NVDA, TSLA, AMZN, META

Update trading bot symbols with these recommendations? (y/N): y
✅ Trading bot symbols updated successfully!
📄 Research report saved to research_report_20241201_143022.json
```

## 📁 File Structure

```
trading_bot/
├── research/                           # New AI research module
│   ├── __init__.py
│   └── langgraph_trading_workflow.py   # Main workflow implementation
├── update_symbols_with_research.py     # Standalone research script
├── start.sh                           # Updated with 'research' command
├── requirements.txt                   # Added LangGraph dependencies
└── LANGGRAPH_INTEGRATION.md          # This documentation
```

## 🔑 Key Features

### 1. **Intelligent Position Analysis**
- Fetches real Alpaca positions via API
- Sorts by market value (descending) 
- Limits to 14 positions (Alpaca paper trading limit)
- Analyzes current holdings vs new opportunities

### 2. **AI-Powered Market Research**
- Multi-query web search for trending stocks
- Real-time market data collection
- LLM-powered trend analysis
- Confidence scoring for recommendations

### 3. **Smart Symbol Extraction**
- Uses advanced LLM to parse market research
- Identifies actual stock ticker symbols
- Filters for quality and relevance
- Fallback mechanisms for reliability

### 4. **Risk-Aware Recommendations**
- Considers existing portfolio composition
- Avoids duplicate positions
- Maintains position limits
- Provides confidence metrics

### 5. **Seamless Integration**
- Automatic `.env` file updates
- Backup creation before changes
- Detailed research reports
- User confirmation before updates

## 🔧 Configuration

### Environment Variables

The integration uses existing trading bot configuration:

```bash
# Alpaca Configuration  
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Symbols (updated by research workflow)
SYMBOLS=AAPL,GOOGL,MSFT,NVDA,TSLA,AMZN,META
```

### LangGraph Cluster Health Check

The script automatically tests cluster connectivity:

```bash
./start.sh research
# Automatically checks:
# ✅ Jetson LLM: Connected
# ✅ Tools Server: Connected  
# ✅ Embeddings: Connected
```

## 📊 Research Reports

Each research session generates detailed JSON reports:

```json
{
  "timestamp": "2024-12-01T14:30:22.123456",
  "success": true,
  "final_symbols": ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"],
  "confidence_scores": {
    "NVDA": 0.8,
    "TSLA": 0.8,
    "AMZN": 0.7
  },
  "current_positions": [...],
  "market_research": "...",
  "analyzed_symbols": [...],
  "error": ""
}
```

## 🚨 Troubleshooting

### Common Issues

1. **LangGraph Cluster Not Accessible**
   ```bash
   # Check cluster status
   cd /home/sanzad/git/langgraph
   python3 cluster_orchestrator.py status
   
   # Start cluster if needed
   python3 cluster_orchestrator.py start
   ```

2. **Alpaca API Errors**
   ```bash
   # Verify credentials in .env
   grep ALPACA .env
   
   # Test API connectivity
   curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
        -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
        https://paper-api.alpaca.markets/v2/account
   ```

3. **Missing Dependencies**
   ```bash
   # Install manually if needed
   pip install langgraph langchain aiohttp
   ```

4. **Network Connectivity**
   ```bash
   # Test individual endpoints
   curl http://192.168.1.177:11434/api/tags      # Jetson LLM
   curl http://192.168.1.190:8082/               # Tools Server
   curl http://192.168.1.81:9002/                # Embeddings
   ```

### Performance Optimization

- **Fast Research**: Uses Jetson for quick analysis
- **Deep Analysis**: Uses CPU node for complex reasoning
- **Parallel Processing**: Concurrent API calls where possible
- **Caching**: Research reports saved for reference
- **Error Handling**: Graceful fallbacks and recovery

## 🎯 Benefits

### For Learning & Development

1. **Real AI Integration**: Hands-on experience with distributed AI systems
2. **Market Intelligence**: Learn how AI can analyze financial markets
3. **Workflow Orchestration**: Understanding LangGraph patterns
4. **API Integration**: Practical experience with financial APIs

### For Trading

1. **Automated Research**: No manual symbol selection needed
2. **Data-Driven Decisions**: AI-powered market analysis
3. **Risk Management**: Intelligent position sizing
4. **Time Savings**: Automated trend identification

### For Infrastructure

1. **Resource Utilization**: Makes full use of your hardware
2. **Scalable Design**: Easy to extend with new analysis types
3. **Local Processing**: Complete privacy and control
4. **Cost Effective**: Zero external API costs

## 🔄 Workflow Customization

You can customize the research workflow by modifying:

1. **Search Queries**: Edit `search_queries` in `market_research_node`
2. **Analysis Prompts**: Customize LLM prompts for different strategies
3. **Symbol Limits**: Adjust the number of symbols analyzed
4. **Confidence Thresholds**: Change recommendation criteria
5. **Risk Parameters**: Modify position sizing logic

## 🚀 Next Steps

1. **Run Research**: `./start.sh research` 
2. **Review Results**: Check generated reports
3. **Start Trading**: `./start.sh run` with new symbols
4. **Monitor Performance**: Use `./start.sh web` dashboard
5. **Iterate**: Re-run research periodically for new opportunities

---

**Happy AI-Enhanced Trading!** 🤖📈

This integration represents a significant advancement in automated trading research, combining the power of your local AI cluster with professional trading infrastructure.
