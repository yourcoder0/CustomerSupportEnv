with open("inference.py", "rb") as f:
    data = f.read()

clean = data.decode("utf-8", errors="ignore")

with open("inference.py", "w", encoding="utf-8") as f:
    f.write(clean)

print("✅ inference.py cleaned successfully")