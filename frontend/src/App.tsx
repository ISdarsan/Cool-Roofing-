import { Component, lazy, Suspense } from "react";
import type { ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { ErrorState, LoadingBlock } from "./components/ui";
import Overview from "./pages/Overview";
import Predict from "./pages/Predict";

// Heavier, chart-rich pages are code-split.
const Compare = lazy(() => import("./pages/Compare"));
const Simulator = lazy(() => import("./pages/Simulator"));
const Performance = lazy(() => import("./pages/Performance"));
const Insights = lazy(() => import("./pages/Insights"));
const Proposal = lazy(() => import("./pages/Proposal"));
const Assistant = lazy(() => import("./pages/Assistant"));

class Boundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) return <ErrorState error={this.state.error} title="This page failed to render" onRetry={() => this.setState({ error: null })} />;
    return this.props.children;
  }
}

export default function App() {
  return (
    <Boundary>
      <Suspense fallback={<div className="p-6"><LoadingBlock label="Loading page…" /></div>}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Overview />} />
            <Route path="predict" element={<Predict />} />
            <Route path="compare" element={<Compare />} />
            <Route path="simulate" element={<Simulator />} />
            <Route path="performance" element={<Performance />} />
            <Route path="insights" element={<Insights />} />
            <Route path="proposal" element={<Proposal />} />
            <Route path="assistant" element={<Assistant />} />
            <Route path="*" element={<ErrorState error={new Error("That page does not exist.")} title="Page not found" />} />
          </Route>
        </Routes>
      </Suspense>
    </Boundary>
  );
}
