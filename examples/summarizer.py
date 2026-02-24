import drift_runtime
from dataclasses import dataclass

@dataclass
class PageSummary:
    title: str
    summary: str
    key_points: list[str]
    word_count: float
    reading_time: str
    tone: str
url = "https://paulgraham.com/writes.html"
print(f"Fetching: {url}")
print("")
content = drift_runtime.fetch(url)
analysis = drift_runtime.ai.ask("Analyze this web page content. Provide a clear summary, 3-5 key points, estimate the word count, calculate reading time, and describe the overall tone.", schema=PageSummary, context={"page_content": content})
print("================================")
print("  PAGE SUMMARY")
print("================================")
print("")
print(f"Title:        {analysis.title}")
print(f"Reading Time: {analysis.reading_time}")
print(f"Tone:         {analysis.tone}")
print(f"Word Count:   ~{analysis.word_count}")
print("")
print("SUMMARY:")
print(f"{analysis.summary}")
print("")
print("KEY POINTS:")
for point in analysis.key_points:
    print(f"  - {point}")
drift_runtime.save(analysis, "page_summary.json")
print("")
print("Full analysis saved to page_summary.json")
