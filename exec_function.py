import io
import sys

old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()

code = """
import random
print(random.randint(1,10)) 
"""

try:
    exec(code)
except Exception as ex:
    print(f"Error executing code: {ex}")

sys.stdout = old_stdout
output = buffer.getvalue()

print(output)
