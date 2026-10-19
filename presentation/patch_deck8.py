p = r"D:\bia hackathon\presentation\build_deck8.py"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert a in s, a[:70]
    s = s.replace(a, b, 1)


# slide 3
rep('s = slide(3, "A first audit found hidden placeholders and fixed them without deleting data"', 's = slide(3, "The audit found hidden placeholders; no data was deleted"')
rep('ax.invert_yaxis(); ax.set_xlim(0, 215); ax.set_xlabel("records / cells")', 'ax.invert_yaxis(); ax.set_xlim(0, 215); ax.set_xticks([0, 50, 100, 150, 200]); ax.set_xlabel("records / cells")')
rep("     size=12, tsize=15)\npic(s, \"d8_anomalies.png\"", "     size=13, tsize=15)\npic(s, \"d8_anomalies.png\"")
# slide 6 title + classification table
rep('s = slide(6, "Cost, cooling and savings are predicted accurately; rating and referral are not"', 's = slide(6, "Strong on cost, cooling and savings; weak on rating and referral"')
i = s.index("verd = {")
j = s.index("card(s, 8.95, 3.9, 3.75, 1.55")
new_tbl = '''short = {"better": "better", "worse": "worse", "not_distinguishable": "n.s."}
crows = [["Task", "Our model", "Trivial baseline", "Verdict"],
         ["Rating (MAE)", f"{ra['test']['mae']:.2f} ★ · R² {ra['test']['r2']:.2f}", f"{ra['test_baseline']['mae']:.2f} ★ (median)", "better"],
         ["High rating ≥4★", f"acc {hr['test']['accuracy']:.1%} · F1 {hr['test']['f1']:.3f} · AUC {hr['test']['roc_auc']:.2f}", f"acc {hr['baseline_accuracy']:.1%} · F1 {hr['baseline_f1']:.3f}", f"acc {short[sg_h['accuracy_verdict']]} · F1 {short[sg_h['f1_verdict']]}"],
         ["Referral", f"F1 {rfr['test']['f1']:.3f} · acc {rfr['test']['accuracy']:.1%} · AUC {rfr['test']['roc_auc']:.2f}", f"F1 {rfr['baseline_f1']:.3f} · acc {rfr['baseline_accuracy']:.1%}", f"F1 {short[sg_r['f1_verdict']]} · acc {short[sg_r['accuracy_verdict']]}"]]
table(s, 0.6, 5.55, crows, [1.5, 3.05, 2.1, 1.5], size=11, rowh=0.36)
'''
s = s[:i] + new_tbl + s[j:]
rep("'Better' only if the 95% bootstrap interval excludes 0.\"", "'Better' only if the 95% bootstrap interval excludes 0; n.s. = not statistically distinguishable.\"")
# slide 7 captions
rep('("d8_predict.png", "Customer prediction", "cost, cooling, kWh, rating, referral + range")', '("d8_predict.png", "Customer prediction", "cost, cooling, kWh, rating, referral")')
rep('("d8_assistant.png", "AI assistant (optional)", "Gemini live-tested; Claude not")', '("d8_assistant.png", "AI assistant", "Gemini live-tested; Claude untested")')
open(p, "w", encoding="utf-8").write(s)
print("patched")
