import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { AppProvider } from "../state/AppState";
import * as fx from "./fixtures";

type Handler = (body: unknown) => { status?: number; json: unknown };
let handlers: Record<string, Handler>;
let calls: { url: string; body: unknown }[];

function installFetch() {
  calls = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const path = String(url).replace(/^\/api/, "");
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url: path, body });
    const h = handlers[path];
    if (!h) return new Response(JSON.stringify({ detail: `no mock for ${path}` }), { status: 404 });
    const r = h(body);
    return new Response(JSON.stringify(r.json), { status: r.status ?? 200, headers: { "Content-Type": "application/json" } });
  }));
}

beforeEach(() => {
  sessionStorage.clear();
  handlers = {
    "/status": () => ({ json: fx.status }), "/schema": () => ({ json: fx.schema }), "/predict": () => ({ json: fx.predict }),
    "/compare": () => ({ json: fx.compare }), "/models": () => ({ json: fx.models }), "/data-quality": () => ({ json: fx.dataQuality }),
    "/insights": () => ({ json: fx.insights }),
  };
  installFetch();
});

function renderAt(path: string) {
  return render(<MemoryRouter initialEntries={[path]}><AppProvider><App /></AppProvider></MemoryRouter>);
}
const seedForm = (over: Record<string, string> = {}) => sessionStorage.setItem("coolroof.form.v1", JSON.stringify({ ...fx.FORM_OK, ...over }));

describe("navigation", () => {
  it("shows every main section and moves between pages", async () => {
    const user = userEvent.setup();
    renderAt("/");
    const nav = await screen.findByRole("navigation", { name: "Main" });
    for (const label of ["Overview", "Customer prediction", "Roofing comparison", "Model performance", "Data quality & insights", "Proposal", "AI assistant"])
      expect(within(nav).getByRole("link", { name: label })).toBeInTheDocument();
    await user.click(within(nav).getByRole("link", { name: "Customer prediction" }));
    expect(await screen.findByRole("heading", { name: "Customer prediction", level: 1 })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main");
  });

  it("unknown route shows a not-found state, not a blank page", async () => {
    renderAt("/nope");
    expect(await screen.findByText("Page not found")).toBeInTheDocument();
  });
});

describe("overview", () => {
  it("renders live status values from the API", async () => {
    renderAt("/");
    expect(await screen.findByText(/Models ready/)).toBeInTheDocument();
    expect(screen.getAllByText(/tested on 500 unseen buildings/).length).toBeGreaterThan(0);
    expect(screen.getByText(/typical errors/)).toBeInTheDocument();
  });

  it("shows an offline error instead of invented numbers when the service is down", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("network"); }));
    renderAt("/");
    expect(await screen.findAllByText(/Prediction service unreachable/)).not.toHaveLength(0);
    expect(screen.queryByText(/Typical miss/)).not.toBeInTheDocument();
    expect(screen.getByText(/Models offline/)).toBeInTheDocument();
  });
});

describe("customer prediction", () => {
  it("starts with an empty state and no prediction call", async () => {
    renderAt("/predict");
    expect(await screen.findByText("No prediction yet")).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/predict")).toBe(false);
  });

  it("validates inputs: required size, out-of-range values, and focuses an error summary", async () => {
    const user = userEvent.setup();
    renderAt("/predict");
    await screen.findByText("No prediction yet");
    await user.type(screen.getByLabelText(/Building age/), "250");
    await user.click(screen.getByRole("button", { name: /Predict outcomes/ }));
    const summary = await screen.findByRole("alert", { name: /Please fix 2 things/ });
    expect(summary).toHaveFocus();
    expect(within(summary).getByText("Building size is required.")).toBeInTheDocument();
    expect(within(summary).getByText(/Building age must be between 0 and 100/)).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/predict")).toBe(false);          // nothing sent while invalid
  });

  it("submits and renders every result with ranges, reliability and probabilities", async () => {
    const user = userEvent.setup();
    renderAt("/predict");
    await user.click(await screen.findByRole("button", { name: /Load example/ }));
    await user.click(screen.getByRole("button", { name: /Predict outcomes/ }));
    expect(await screen.findByTestId("cost-value")).toHaveTextContent("₹15,00,000");     // Indian digit grouping
    expect(screen.getByTestId("temp-value")).toHaveTextContent("9.5");
    expect(screen.getByTestId("units-value")).toHaveTextContent("100");
    expect(screen.getByTestId("rating-value")).toHaveTextContent("4.1");
    expect(screen.getByTestId("high-value")).toHaveTextContent("77%");
    expect(screen.getByTestId("referral-value")).toHaveTextContent("58%");
    expect(screen.getAllByText(/likely range/).length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText(/Low reliability/)).toBeInTheDocument();
    expect(screen.getByText(/filled with typical training values/)).toBeInTheDocument();      // warning surfaced
    const sent = calls.find((c) => c.url === "/predict")!.body as { building: Record<string, unknown> };
    expect(sent.building).toMatchObject({ building_size: 1500, num_ac_units: 2, cool_roofing_method: "sheet_with_coating" });
    expect(Object.keys(sent.building)).not.toContain("cool_roofing_cost");
  });

  it("distinguishes the expected rating from the probability of a high rating", async () => {
    const user = userEvent.setup();
    renderAt("/predict");
    await user.click(await screen.findByRole("button", { name: /Load example/ }));
    await user.click(screen.getByRole("button", { name: /Predict outcomes/ }));
    await screen.findByTestId("cost-value");
    expect(screen.getByText(/It is not the probability below/)).toBeInTheDocument();
    expect(screen.getByText(/assumed; organisers gave no definition/)).toBeInTheDocument();
  });

  it("shows a readable error with retry when the service returns 503, and no fake values", async () => {
    handlers["/predict"] = () => ({ status: 503, json: { detail: "Models have not been trained yet." } });
    const user = userEvent.setup();
    renderAt("/predict");
    await user.click(await screen.findByRole("button", { name: /Load example/ }));
    await user.click(screen.getByRole("button", { name: /Predict outcomes/ }));
    expect(await screen.findByText("Models not ready")).toBeInTheDocument();
    expect(screen.getByText(/Models have not been trained yet/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Try again/ })).toBeInTheDocument();
    expect(screen.queryByTestId("cost-value")).not.toBeInTheDocument();
  });

  it("only offers methods the history supports for the chosen roof type", async () => {
    const user = userEvent.setup();
    renderAt("/predict");
    await screen.findByText("No prediction yet");
    await waitFor(() => expect(screen.getByRole("radio", { name: /Tiles \+ coating/ })).toBeInTheDocument());
    await user.selectOptions(screen.getByLabelText("Existing roof"), "sheet");
    await waitFor(() => expect(screen.queryByRole("radio", { name: /Tiles \+ coating/ })).not.toBeInTheDocument());
    expect(screen.getByRole("radio", { name: /Sheet \+ coating/ })).toBeInTheDocument();
  });

  it("computes payback only from a tariff the user types (never assumed)", async () => {
    handlers["/economics"] = () => ({ json: { inputs: {}, central: { simple_payback_years: 12.5, annual_bill_saving_inr: 9600, breakeven_tariff_inr_per_kwh: 100, monthly_kwh_saved: 100, annual_net_saving_inr: 9600, payback_with_escalation_years: null, net_position_after_horizon_inr: 0 }, range: null, assumptions: ["Estimate, not a guarantee"], warnings: [] } });
    const user = userEvent.setup();
    renderAt("/predict");
    await user.click(await screen.findByRole("button", { name: /Load example/ }));
    await user.click(screen.getByRole("button", { name: /Predict outcomes/ }));
    await screen.findByTestId("cost-value");
    expect(screen.getByRole("spinbutton", { name: "Tariff" })).toHaveValue(null);                         // empty by default
    await user.click(screen.getByRole("button", { name: /Calculate payback/ }));
    expect(await screen.findByText(/never assumed/)).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/economics")).toBe(false);
    await user.type(screen.getByRole("spinbutton", { name: "Tariff" }), "8");
    await user.click(screen.getByRole("button", { name: /Calculate payback/ }));
    expect(await screen.findByTestId("payback-years")).toHaveTextContent("12.5 years");
    expect((calls.find((c) => c.url === "/economics")!.body as Record<string, number>).tariff_inr_per_kwh).toBe(8);
  });
});

describe("roofing comparison", () => {
  it("asks for building details first when there are none", async () => {
    renderAt("/compare");
    expect(await screen.findByText("Enter the building details first")).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/compare")).toBe(false);
  });

  it("compares options, explains the recommendation and lets the user adopt it", async () => {
    seedForm();
    const user = userEvent.setup();
    renderAt("/compare");
    expect(await screen.findByRole("heading", { name: "Tiles + coating · High" })).toBeInTheDocument();
    expect(screen.getByText(/Weighted sum of min–max normalised metrics/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /Predicted outcomes for every supported roofing option/ })).toBeInTheDocument();
    expect(screen.getAllByText("current")).toHaveLength(1);
    expect(screen.getByText("Winner by single objective")).toBeInTheDocument();
    expect(calls.find((c) => c.url === "/compare")!.body).toMatchObject({ objective: "balanced", current_method: "sheet_with_coating", current_quality: "high" });
    await user.click(screen.getByRole("button", { name: "Use this option" }));
    expect(await screen.findByText(/Selected for this customer/)).toBeInTheDocument();
  });

  it("re-requests with the chosen objective and hides weights for single-metric objectives", async () => {
    seedForm();
    const user = userEvent.setup();
    renderAt("/compare");
    await screen.findByRole("heading", { name: "Tiles + coating · High" });
    expect(screen.getByRole("heading", { name: "Priorities" })).toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /Lowest estimated installation cost/ }));
    await waitFor(() => expect(calls.filter((c) => c.url === "/compare").some((c) => (c.body as { objective: string }).objective === "cost")).toBe(true));
    expect(screen.queryByRole("heading", { name: "Priorities" })).not.toBeInTheDocument();
  });

  it("shows an error state when comparison fails", async () => {
    seedForm();
    handlers["/compare"] = () => ({ status: 422, json: { detail: "No supported roofing methods exist for this roof type." } });
    renderAt("/compare");
    expect(await screen.findByText(/No supported roofing methods/)).toBeInTheDocument();
  });
});

describe("assistant (optional)", () => {
  it("explains how to enable it and does not offer a chat box when disabled", async () => {
    renderAt("/assistant");
    expect(await screen.findByText("The assistant is not available")).toBeInTheDocument();
    expect(screen.getByText(/COOLROOF_LLM_PROVIDER=builtin/, { selector: "code" })).toBeInTheDocument();
    expect(screen.queryByLabelText(/Message to the assistant/)).not.toBeInTheDocument();
  });
});

describe("proposal", () => {
  it("needs a building first", async () => {
    renderAt("/proposal");
    expect(await screen.findByText("No building to propose for yet")).toBeInTheDocument();
  });

  it("builds the PDF request from the real inputs and rejects an invalid tariff locally", async () => {
    seedForm();
    handlers["/proposal/pdf"] = () => ({ json: {} });
    const user = userEvent.setup();
    renderAt("/proposal");
    await screen.findByText("What the proposal will say");
    await user.type(screen.getByLabelText(/Electricity tariff/), "500");
    await user.click(screen.getByRole("button", { name: /Download PDF proposal/ }));
    expect(await screen.findByText(/Tariff must be between 0 and 100/)).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/proposal/pdf")).toBe(false);
  });
});

describe("accessibility (axe)", () => {
  it("overview has no detectable violations", async () => {
    const { container } = renderAt("/");
    await screen.findByText(/Models ready/);
    await screen.findAllByText(/tested on 500 unseen buildings/);
    expect(await axe(container)).toHaveNoViolations();
  });

  it("prediction form and results have no detectable violations", async () => {
    const user = userEvent.setup();
    const { container } = renderAt("/predict");
    await user.click(await screen.findByRole("button", { name: /Load example/ }));
    await user.click(screen.getByRole("button", { name: /Predict outcomes/ }));
    await screen.findByTestId("cost-value");
    expect(await axe(container)).toHaveNoViolations();
  });

  it("comparison page has no detectable violations", async () => {
    seedForm();
    const { container } = renderAt("/compare");
    await screen.findByRole("heading", { name: "Tiles + coating · High" });
    expect(await axe(container)).toHaveNoViolations();
  });

  it("every form control has an accessible name", async () => {
    renderAt("/predict");
    await screen.findByText("No prediction yet");
    for (const el of screen.getAllByRole("spinbutton")) expect(el).toHaveAccessibleName();
    for (const el of screen.getAllByRole("combobox")) expect(el).toHaveAccessibleName();
  });
});


describe("what-if simulator", () => {
  const predictBySize = (body: unknown) => {
    const b = (body as { building: { building_size: number; cool_roofing_quality: string } }).building;
    const r = structuredClone(fx.predict);
    r.predictions.cost.value = b.building_size * 1000 * (b.cool_roofing_quality === "basic" ? 0.6 : 1);   // mock only: varies with the input
    r.predictions.temperature_drop.value = b.cool_roofing_quality === "basic" ? 4 : 9.5;
    return { json: r };
  };

  it("needs a baseline customer first and calls nothing", async () => {
    renderAt("/simulate");
    expect(await screen.findByText("Start from a baseline customer")).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/predict" || c.url === "/compare")).toBe(false);
  });

  it("makes no prediction until Run simulation is pressed", async () => {
    seedForm();
    renderAt("/simulate");
    expect(await screen.findByText("No simulation yet")).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/predict")).toBe(false);
  });

  it("runs baseline + scenario through the real prediction and compare endpoints and shows both side by side", async () => {
    seedForm();
    handlers["/predict"] = predictBySize;
    const user = userEvent.setup();
    renderAt("/simulate");
    await screen.findByText("No simulation yet");
    await user.click(screen.getByRole("radio", { name: "Basic" }));                      // scenario: quality high -> basic
    const size = screen.getByRole("spinbutton", { name: "Building size" });
    await user.clear(size);
    await user.type(size, "2000");
    expect(screen.getByText(/2 changes from the baseline/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Run simulation/ }));
    expect(await screen.findByTestId("sim-scen-cost")).toBeInTheDocument();

    const predicts = calls.filter((c) => c.url === "/predict").map((c) => (c.body as { building: Record<string, unknown> }).building);
    expect(predicts).toHaveLength(2);
    expect(predicts[0]).toMatchObject({ building_size: 1500, cool_roofing_quality: "high" });          // baseline untouched
    expect(predicts[1]).toMatchObject({ building_size: 2000, cool_roofing_quality: "basic" });         // scenario
    const cmp = calls.find((c) => c.url === "/compare")!.body as { building: Record<string, unknown>; current_quality: string };
    expect(cmp.building).toMatchObject({ building_size: 2000 });
    expect(cmp.current_quality).toBe("basic");
    // values on screen are exactly what the (mocked) API returned: 1500*1000 baseline, 2000*1000*0.6 scenario
    expect(screen.getByTestId("sim-base-cost")).toHaveTextContent("₹15,00,000");
    expect(screen.getByTestId("sim-scen-cost")).toHaveTextContent("₹12,00,000");
    expect(screen.getByTestId("sim-delta-cost")).toHaveTextContent(/−.*20%/);
    expect(screen.getByTestId("sim-base-temperature_drop")).toHaveTextContent("9.5");
    expect(screen.getByTestId("sim-scen-temperature_drop")).toHaveTextContent("4.0");
    expect(screen.getByText(/Changed:/)).toHaveTextContent(/Quality grade High → Basic/);
    expect(screen.getByText("Roofing options for this scenario's building")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Savings" })).toBeInTheDocument();
  });

  it("flags a scenario identical to the baseline instead of implying an effect", async () => {
    seedForm();
    const user = userEvent.setup();
    renderAt("/simulate");
    await user.click(await screen.findByRole("button", { name: /Run simulation/ }));
    expect(await screen.findByText(/scenario is identical to the baseline/)).toBeInTheDocument();
  });

  it("validates scenario inputs before calling the API", async () => {
    seedForm();
    const user = userEvent.setup();
    renderAt("/simulate");
    const size = await screen.findByRole("spinbutton", { name: "Building size" });
    await user.clear(size);
    await user.click(screen.getByRole("button", { name: /Run simulation/ }));
    expect(await screen.findByText("Building size is required.")).toBeInTheDocument();
    expect(calls.some((c) => c.url === "/predict")).toBe(false);
  });

  it("only offers methods supported for the scenario's roof type", async () => {
    seedForm();
    const user = userEvent.setup();
    renderAt("/simulate");
    await screen.findByText("No simulation yet");
    expect(screen.getByRole("radio", { name: "Tiles + coating" })).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Existing roof"), "sheet");
    await waitFor(() => expect(screen.queryByRole("radio", { name: "Tiles + coating" })).not.toBeInTheDocument());
  });

  it("shows an error and no fake numbers when the service fails", async () => {
    seedForm();
    handlers["/predict"] = () => ({ status: 503, json: { detail: "Models have not been trained yet." } });
    const user = userEvent.setup();
    renderAt("/simulate");
    await user.click(await screen.findByRole("button", { name: /Run simulation/ }));
    expect(await screen.findByText("Models not ready")).toBeInTheDocument();
    expect(screen.queryByTestId("sim-scen-cost")).not.toBeInTheDocument();
  });

  it("is reachable from the navigation and has no detectable accessibility violations", async () => {
    seedForm();
    handlers["/predict"] = predictBySize;
    const user = userEvent.setup();
    const { container } = renderAt("/");
    await user.click(await screen.findByRole("link", { name: "What-if simulator" }));
    await user.click(await screen.findByRole("button", { name: /Run simulation/ }));
    await screen.findByTestId("sim-scen-cost");
    expect(await axe(container)).toHaveNoViolations();
  });
});
