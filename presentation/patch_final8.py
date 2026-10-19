from pathlib import Path

R = Path(r"D:\bia hackathon\presentation")


def edit(name, pairs):
    p = R / name
    s = p.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (name, a[:60])
        s = s.replace(a, b, 1)
    p.write_text(s, encoding="utf-8")


edit("build_deck8.py", [("ax.set_xlim(0, 215); ax.set_xticks([0, 50, 100, 150, 200])", "ax.set_xlim(0, 230); ax.set_xticks([0, 100, 200])")])
edit("make_fact_sheet.py", [
    ("- Test count: the brief says 107 backend tests; the project now has **153** (new review, assistant-provider and significance tests were added). Frontend: now 30 (22 at the time of the brief, plus 8 what-if simulator tests).",
     "- Test counts: **153 backend** and **30 frontend** tests. Older figures of 107 / 22 (seen in earlier briefs) are out of date: review, assistant-provider, significance and what-if simulator tests were added."),
    ("Slides were rendered with Microsoft PowerPoint itself (PNG per slide and a PDF) and inspected from contact sheets and full-size renders; see the final report for what was and was not checked.",
     "The 8-slide deck was rendered with Microsoft PowerPoint itself (PNG per slide plus a PDF in `presentation_assets/render8/` and `Cool_Roofing_Hackathon_Presentation_8slides.pdf`) and inspected from contact sheets and full-size renders. "
     "The canonical file name is currently held open (locked) by the user's PowerPoint, so the new deck was written to `Cool_Roofing_Hackathon_Presentation_8slides.pptx`; the earlier 15-slide deck is preserved in `presentation/archive_15_slide/`."),
])
print("ok")
