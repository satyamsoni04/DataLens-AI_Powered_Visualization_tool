# DataLens

A browser-based data analytics and visualization platform. Upload CSV or Excel datasets, automatically analyze column types and dataset health, clean your data, generate interactive visualizations, get statistical insights, and export results — all without leaving your browser. No data ever leaves your machine.

## Features

- **Dataset Upload** — Drag-and-drop CSV and Excel file import with automatic parsing
- **Automated Dataset Analysis** — Column type detection (numeric, categorical, date, boolean, text) and dataset health scoring
- **Data Cleaning Tools** — Remove duplicates, fill missing values, cap or remove outliers
- **Interactive Visualizations** — Auto-generated Plotly.js charts (line, bar, pie, histogram, box plots) with hover details and zoom
- **Export Options** — Download cleaned data as CSV or Excel, export charts as PNG (or ZIP bundle), generate full PDF analytics reports
- **Analytics Report Generation** — PDF reports with cover page, dataset overview, insights summary, chart images, and sample data table
- **Dark / Light Mode** — Toggle between themes with smooth transitions; preference saved to localStorage
- **Resizable Dashboard Panels** — Drag handles between sidebar, charts, and data table to customize your workspace
- **Fullscreen Chart and Table Views** — Expand any chart to focus mode or make the data table fullscreen for detailed exploration
- **Collapsible Sidebar** — Collapse the sidebar to an icon rail to maximize workspace area
- **Column Inspector** — Click any column to see detailed statistics and mini-visualizations in a side panel
- **Statistical Insights** — Auto-generated insights highlighting trends, correlations, and anomalies

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Shadcn UI
- **Charts**: Plotly.js via react-plotly.js
- **CSV Parsing**: Papa Parse
- **Excel**: ExcelJS
- **PDF Reports**: jsPDF + jspdf-autotable
- **Resizable Panels**: react-resizable-panels
- **Backend**: Express.js (minimal — only serves the frontend in production)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/DataLens.git
cd DataLens
```

2. Install dependencies and start the development server:

```bash
npm install
npm run dev
```

The app will be available at (https://data-lens-ai-visualization-satyamsoni.lovable.app)

## Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
DataLens/
├── client/
│   ├── src/
│   │   ├── components/       # UI components (Navbar, ChartGrid, DatasetTable, etc.)
│   │   │   └── ui/           # Shadcn UI primitives
│   │   ├── hooks/            # Custom React hooks
│   │   ├── lib/              # Core engines (parsing, analysis, cleaning, charts, export)
│   │   └── pages/            # Page components (Dashboard)
│   ├── public/               # Static assets
│   └── index.html            # Entry HTML
├── server/                   # Express server (serves frontend, minimal API)
├── shared/                   # Shared TypeScript schemas
├── package.json
├── tailwind.config.ts
├── vite.config.ts
├── tsconfig.json
└── README.md
```

## Privacy

All data processing happens entirely in your browser. No data is uploaded to any server. No external APIs are called. Your datasets stay on your machine.

## License

MIT License — see [LICENSE](LICENSE) for details.
