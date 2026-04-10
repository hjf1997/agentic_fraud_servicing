import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./App";
import DemoPage from "./DemoPage";
import LayersPage from "./LayersPage";
import AgentDiagramPage from "./AgentDiagramPage";
import ExportWorkflow from "./ExportWorkflow";
import VisionPage from "./VisionPage";

function Router() {
  const [page, setPage] = useState(window.location.hash);

  useEffect(() => {
    const onHash = () => setPage(window.location.hash);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  if (page === "#/export") return <ExportWorkflow />;
  if (page === "#/vision") return <VisionPage />;
  if (page === "#/demo") return <DemoPage />;
  if (page === "#/layers") return <LayersPage />;
  if (page.startsWith("#/agent")) return <AgentDiagramPage />;
  return <App />;
}

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);
root.render(
  <React.StrictMode>
    <Router />
  </React.StrictMode>
);
