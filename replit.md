# DataLens - Browser-Based Data Analytics Platform

## Overview
DataLens is a browser-based data analytics and visualization platform. Users can upload datasets (CSV/Excel), automatically analyze them, detect data quality issues, clean the data, and generate interactive visualizations and insights. All processing happens client-side in the browser.

## Architecture
- **Frontend**: React + TypeScript with Vite, Tailwind CSS, Shadcn UI
- **Backend**: Express.js (minimal - only serves the frontend)
- **Charts**: Plotly.js via react-plotly.js
- **CSV Parsing**: Papa Parse
- **Excel**: SheetJS (xlsx)
- **Export**: Built-in browser download (CSV), ExcelJS (Excel), JSZip (chart PNG bundles)
- **PDF Reports**: jsPDF + jspdf-autotable
- **Layout**: react-resizable-panels for resizable dashboard panels

## Project Structure

### Client-Side Engines (`client/src/lib/`)
- `types.ts` - TypeScript types for columns, datasets, charts, insights
- `dataParser.ts` - CSV/Excel parsing, column type detection
- `dataAnalysis.ts` - Dataset health calculation, correlation matrix
- `cleaningEngine.ts` - Data cleaning operations (duplicates, missing values, outliers)
- `chartEngine.ts` - Automatic chart generation based on column types
- `insightsEngine.ts` - Statistical insight generation
- `exportEngine.ts` - CSV (built-in), Excel (ExcelJS), Charts as PNG (Plotly.toImage + JSZip)
- `reportEngine.ts` - PDF analytics report generation (cover page, overview, insights, charts, sample table)
- `sampleData.ts` - Built-in sample dataset
- `theme.tsx` - ThemeProvider context, toggle, chart theme layout helpers
- `store.ts` - React hook managing all application state

### Components (`client/src/components/`)
- `Navbar.tsx` - Top navigation with upload, export, reset, theme toggle, sidebar toggle
- `ExportModal.tsx` - Export dialog with 4 options: CSV, Excel, Charts PNG, PDF Report
- `DatasetSidebar.tsx` - Left sidebar with upload, health, cleaning tools, fields (collapsible to icon rail)
- `ChartGrid.tsx` - Responsive chart grid with lazy-loaded Plotly charts and expand/focus mode
- `DatasetTable.tsx` - Interactive data table with search, sort, pagination, fullscreen mode
- `ColumnInspector.tsx` - Right panel showing column statistics and mini-charts
- `InsightsPanel.tsx` - Slide-in panel showing auto-generated insights
- `EmptyState.tsx` - Welcome screen with drag-and-drop upload
- `LoadingOverlay.tsx` - Loading indicator during analysis

### Key Features
- CSV and Excel file upload with drag-and-drop
- Automatic column type detection (numeric, categorical, date, boolean, text)
- Dataset health reporting (rows, columns, duplicates, missing values, outliers)
- Data cleaning tools (remove duplicates, fill missing values, cap/remove outliers)
- Automatic chart generation (line, bar, pie, histogram, box plots)
- Column inspector with statistics and mini-visualizations
- Statistical insights engine
- Export modal with 4 options: CSV, Excel, Charts as PNG (ZIP if multiple), PDF analytics report
- PDF report includes cover page, dataset overview, insights, chart images, and sample data table
- Light/Dark mode toggle with localStorage persistence (key: `datalens-theme`)
- Charts, tables, sidebar, modals, and navigation all adapt to theme
- Smooth `transition-colors duration-300` on theme switch
- Resizable dashboard panels (sidebar, charts, table) via drag handles
- Collapsible sidebar (icon-only rail when collapsed, controlled via Panel imperative API)
- Chart expand/focus mode (single chart fills workspace, exit to restore grid)
- Fullscreen data table (hides charts, table fills workspace, search/sort preserved)
- Charts auto-resize via ResizeObserver when panels change size
- Responsive design with sidebar navigation

## No External APIs
All data processing is done client-side. No paid APIs required. No data leaves the browser.
