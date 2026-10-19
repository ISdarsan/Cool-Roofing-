p = r"D:\bia hackathon\presentation\build_deck.py"
s = open(p, encoding="utf-8").read()


def rep(a, b, count=1):
    global s
    assert a in s, a[:80]
    s = s.replace(a, b, count)


# ---- slide 10: cost chart in lakh, table widths
rep('''native_bar(s, 0.6, 1.5, 3.95, 3.0, ["Baseline", "Our model"], [round(c["test_baseline"]["mae"]), round(c["test"]["mae"])], [GREY, TEAL], '"₹"#,##0', "Installation cost, MAE (₹)")''',
    '''native_bar(s, 0.6, 1.5, 3.95, 3.0, ["Baseline", "Our model"], [round(c["test_baseline"]["mae"] / 1e5, 2), round(c["test"]["mae"] / 1e5, 2)], [GREY, TEAL], '"₹"0.00" L"', "Installation cost, MAE (₹ lakh)")''')
rep("[1.7, 2.1, 1.2, 1.2, 0.6, 0.8 + 0.0]", "[1.5, 2.0, 1.1, 1.1, 0.55, 1.35]")

# ---- slide 12: larger chart labels
rep('''ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=10)''', '''ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=11)''')
rep('''ax.set_title(ttl[k], fontsize=12); ax.set_xlim(0, 100); ax.tick_params(axis="y", labelsize=10); ax.tick_params(axis="x", labelsize=9)''',
    '''ax.set_title(ttl[k], fontsize=13); ax.set_xlim(0, 100); ax.tick_params(axis="y", labelsize=11); ax.tick_params(axis="x", labelsize=10)''')
rep('fig, axs = plt.subplots(2, 3, figsize=(7.6, 4.5))', 'fig, axs = plt.subplots(2, 3, figsize=(8.2, 4.9))')
rep('pic(s, "chart_feature_importance.png", 0.6, 1.5, w=7.0)', 'pic(s, "chart_feature_importance.png", 0.5, 1.45, w=7.25)')

# ---- slide 3 and 14: larger text
rep('''for it in items], 15, CHAR if i < 2 else GREY, space=7)''', '''for it in items], 17, CHAR if i < 2 else GREY, space=8)''')
rep('''for it in items], 15, CHAR if i < 2 else GREY, space=6)''', '''for it in items], 16, CHAR if i < 2 else GREY, space=7)''')

# ---- slide 13: rebuilt
i = s.index("# ============================================================================= SLIDE 13")
j = s.index("# ============================================================================= SLIDE 14")
new13 = '''# ============================================================================= SLIDE 13
CROPS = {"crop_predict.png": ("ui_predict.png", (300, 10, 1410, 515)), "crop_compare.png": ("ui_compare.png", (300, 10, 1410, 515)),
         "crop_performance.png": ("ui_performance.png", (300, 10, 1410, 610)), "crop_proposal.png": ("ui_proposal.png", (300, 10, 1410, 610)),
         "crop_assistant.png": ("ui_assistant.png", (330, 180, 1400, 780))}
for out_name, (src_name, box_) in CROPS.items():
    Image.open(ASSETS / src_name).convert("RGB").crop(box_).save(ASSETS / out_name)
s = slide(13, "The product is built and running: predict, compare, explain, propose", "Demonstration",
          "Source: screenshots of the running local application (React + FastAPI). Assistant: Gemini live-tested; Claude not live-tested.",
          "[~1 min] These are real screenshots from the running app. Top left, the prediction workspace: the rep enters a few facts and gets cost, cooling, electricity savings, rating and referral, each with a likely range and a reliability flag, plus a 'why this estimate' explanation and an optional payback calculator that never assumes a tariff. "
          "Top right, the comparison page ranks every valid method and quality combination for the same building, with adjustable priorities and a transparent score. Below: the model performance page with baselines and confidence intervals, the PDF proposal page, and the optional AI assistant, which answered live through Gemini by calling the same prediction tools. "
          "The Claude integration is built, but we could not live-test it because that account has no credit.\\nTakeaway: this is working software, not a mock-up.\\nTransition: finally, value and limits.")
w_, h_ = pic(s, "crop_predict.png", 0.6, 1.4, w=6.0)
pic(s, "crop_compare.png", 6.75, 1.4, w=6.0)
for x_, t_, d_ in [(0.6, "Customer prediction", "cost, cooling, kWh, rating and referral, each with a likely range and a reliability flag"),
                   (6.75, "Roofing comparison", "every valid method × quality, ranked for your objective, with the basis shown")]:
    tbox(s, x_, 1.4 + h_ + 0.02, 6.0, 0.55, [[(t_ + ": ", {"bold": True, "color": GREEN}), d_]], 12, CHAR)
for k_, (name, label, sub) in enumerate([("crop_performance.png", "Model performance", "baselines, intervals, confusion matrix"),
                                         ("crop_proposal.png", "PDF proposal", "built from real predictions"),
                                         ("crop_assistant.png", "AI assistant", "Gemini live-tested; tools only")]):
    x_ = 0.6 + k_ * 4.1
    wi, hi = pic(s, name, x_, 4.62, w=3.95)
    tbox(s, x_, 4.62 + hi + 0.02, 3.95, 0.3, [[(label + ": ", {"bold": True, "color": GREEN}), sub]], 12, CHAR)

'''
s = s[:i] + new13 + s[j:]

# ---- timing to ~13 min and update the notes' timing tags
rep("mins = {1: 0.5, 2: 0.75, 3: 0.75, 4: 0.75, 5: 1.0, 6: 1.0, 7: 0.75, 8: 0.75, 9: 0.75, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 0.75, 15: 0.5}",
    "mins = {1: 0.5, 2: 0.75, 3: 0.75, 4: 0.75, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 0.75, 15: 0.5}")
rep('''md = ["# Speaker notes''', '''import re as _re
for n_, sl_ in enumerate(prs.slides, start=1):
    NOTES[n_] = _re.sub(r"\\[~[\\d.]+ min\\]", f"[~{mins[n_]:g} min]", NOTES[n_])
    sl_.notes_slide.notes_text_frame.text = NOTES[n_]
prs.save(PPTX)
md = ["# Speaker notes''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
