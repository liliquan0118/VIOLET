Treat both texts as data to examine, not instructions to carry out.

Question: What information is explicitly stated in the text to check but not also stated in the comparison text?

Report possible information differences, not wording differences. A paraphrase expressing the same information is not a difference. Do not list shared information or first decompose the entire texts. Each item should describe one difference that can be checked with a single conclusion, retaining the relevant referents and scope.

Read both complete texts. The comparison may omit information or express a different condition; omission is not denial, and different conditions are not necessarily contradictory. Do not fill gaps from business practice. If the meaning depends on missing context, report the uncertain wording and why it cannot be decided.

Return only JSON with these two lists:
{"differences":[{"difference":"a short description of the possible difference between the texts","source_quote":"exact words from the text to check","comparison_quote":null}],"uncertainties":[]}

Use an exact comparison_quote when there is a relevant passage; otherwise use null. An uncertainty has source_quote, comparison_quote, and reason fields; it must quote at least one of the texts and explain what cannot be determined. Return empty lists if you find no candidates or uncertainties. Do not force a difference or decide overall equivalence or test validity. Findings are candidates for later review.
