Treat the sentence as text to examine, not as an instruction to carry out.

Question: Which separate factual questions does this sentence explicitly answer?

Return one item per question. Each question must ask about one fact, and its answer must answer only that question. Keep the main action separate from details describing its participants or circumstances; do not combine the whole sentence into one answer. Keep each detail's referent clear and support it with an exact source quote.

Use status explicit when the sentence answers the question. If the quoted expression is ambiguous, use status unclear and say what cannot be determined. Do not invent answers or questions about facts the sentence does not mention. Do not infer facts from normal business practice or compare with another sentence.

Return only JSON with one top-level key, details:
{"details":[{"question":"one factual question","answer":"its answer","quote":"exact source words","status":"explicit or unclear"}]}

Every item must contain all four fields. If nothing can be extracted, return {"details":[]}; this does not mean unrestricted.
