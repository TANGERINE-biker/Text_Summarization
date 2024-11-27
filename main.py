from transformers import pipeline

# Создаем pipeline для суммаризации
summarizer = pipeline("summarization", framework="tf", model="facebook/bart-large-cnn")

# Текст для суммаризации
text = """
Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans.
The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.
The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal.
Machine learning (ML) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.
Deep learning, on the other hand, is a sub-field of machine learning that uses algorithms inspired by the structure and function of the brain’s neural networks.
"""

# Выполняем суммаризацию
summary = summarizer(text, max_length=100, min_length=25, do_sample=False)

# Выводим результат
print("Original text:")
print(text)
print("\nSummary:")
print(summary[0]['summary_text'])
