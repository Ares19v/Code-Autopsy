import urllib.request
import json
import time

API_URL = "http://localhost:8000/review"

HARD_TEST_CASES = [
    {
        "name": "1. Python Late Binding Closure in List Comprehension",
        "language": "python",
        "code": "def make_multipliers():\n    return [lambda x: i * x for i in range(4)]",
        "expected_bug_keywords": ["late binding", "closure", "scope", "reference", "capture", "i=i", "default", "multiplier"]
    },
    {
        "name": "2. JS Prototype Pollution via Recursive Object Merge",
        "language": "javascript",
        "code": "function merge(target, source) {\n    for (let key in source) {\n        if (typeof source[key] === 'object' && source[key] !== null) {\n            if (!target[key]) target[key] = {};\n            merge(target[key], source[key]);\n        } else {\n            target[key] = source[key];\n        }\n    }\n    return target;\n}",
        "expected_bug_keywords": ["prototype", "pollution", "__proto__", "constructor", "security", "sanitize", "hasownproperty"]
    },
    {
        "name": "3. Python Multithreading Race Condition (Non-Atomic Operation)",
        "language": "python",
        "code": "import threading\ncounter = 0\ndef worker():\n    global counter\n    for _ in range(100000):\n        counter += 1",
        "expected_bug_keywords": ["race condition", "atomic", "thread", "lock", "synchroniz", "mutex", "concurrency"]
    },
    {
        "name": "4. JS React Stale Closure in Custom Hook",
        "language": "javascript",
        "code": "function useInterval(callback, delay) {\n    useEffect(() => {\n        const id = setInterval(callback, delay);\n        return () => clearInterval(id);\n    }, [delay]);\n}",
        "expected_bug_keywords": ["stale", "closure", "dependenc", "callback", "ref", "useref", "useeffect"]
    },
    {
        "name": "5. Python lru_cache Memory Leak on Instance Method",
        "language": "python",
        "code": "from functools import lru_cache\n\nclass DataProcessor:\n    def __init__(self, data):\n        self.data = data\n    @lru_cache(maxsize=128)\n    def process(self, multiplier):\n        return [x * multiplier for x in self.data]",
        "expected_bug_keywords": ["lru_cache", "memory leak", "self", "reference", "garbage", "instance", "method"]
    },
    {
        "name": "6. JS JSON.parse(JSON.stringify) Deep Clone Pitfalls",
        "language": "javascript",
        "code": "function cloneDeep(obj) {\n    return JSON.parse(JSON.stringify(obj));\n}",
        "expected_bug_keywords": ["circular", "date", "function", "undefined", "symbol", "loss", "serialize", "json"]
    },
    {
        "name": "7. JS ReDoS (Regular Expression Denial of Service)",
        "language": "javascript",
        "code": "function validateInput(input) {\n    const regex = /^(a+)+$/;\n    return regex.test(input);\n}",
        "expected_bug_keywords": ["redos", "backtracking", "catastrophic", "exponential", "denial of service", "regex", "performance"]
    },
    {
        "name": "8. Python Subprocess Command Injection via shell=True",
        "language": "python",
        "code": "import subprocess\ndef convert_image(filename):\n    cmd = f\"convert {filename} output.png\"\n    return subprocess.run(cmd, shell=True, check=True)",
        "expected_bug_keywords": ["command injection", "shell=true", "injection", "subprocess", "vulnerability", "shell", "security"]
    },
    {
        "name": "9. JS Array.prototype.sort() In-Place Mutation Side Effect",
        "language": "javascript",
        "code": "function getTopScores(scores) {\n    return scores.sort((a, b) => b - a).slice(0, 3);\n}",
        "expected_bug_keywords": ["mutate", "mutation", "in-place", "original", "side effect", "copy", "slice", "spread"]
    },
    {
        "name": "10. Python Exception Masking with Bare Except / Pass",
        "language": "python",
        "code": "def parse_payload(data):\n    try:\n        return json.loads(data)\n    except:\n        pass",
        "expected_bug_keywords": ["bare except", "silently", "mask", "suppress", "exception", "pass", "jsondecodeerror"]
    }
]

def run_tests():
    print("="*65)
    print("🧠 Code Autopsy - Hard & Nuanced Problems Benchmark")
    print("="*65)
    
    passed = 0
    warnings = 0
    
    for i, test in enumerate(HARD_TEST_CASES):
        print(f"\n─────────────────────────────────────────────────────────────")
        print(f"▶ Hard Test {i+1}: {test['name']} ({test['language']})")
        print(f"Code Snippet:\n{test['code']}")
        print(f"─────────────────────────────────────────────────────────────")
        
        req_data = json.dumps({
            "code": test["code"],
            "language": test["language"]
        }).encode("utf-8")
        
        req = urllib.request.Request(API_URL, data=req_data, headers={"Content-Type": "application/json"})
        
        try:
            t0 = time.time()
            res = urllib.request.urlopen(req)
            t1 = time.time()
            
            response_data = json.loads(res.read().decode())
            
            bug_identified = response_data.get("bug_identified", "").lower()
            root_cause = response_data.get("root_cause", "").lower()
            fixed_code = response_data.get("fixed_code", "")
            confidence = response_data.get("confidence", 0.0)
            combined_analysis = bug_identified + " " + root_cause
            
            if "no critical bug" in combined_analysis or "no specific issue" in combined_analysis:
                status = "❌ FAILED (Model claimed 'No bug found')"
            else:
                matched_keywords = [kw for kw in test["expected_bug_keywords"] if kw in combined_analysis]
                if matched_keywords:
                    status = f"✅ PASSED (Matched keywords: {matched_keywords})"
                    passed += 1
                else:
                    status = "⚠️ WARNING (Found a bug, but matched keyword set differs)"
                    warnings += 1
            
            print(f"Status: {status}")
            print(f"Latency: {round((t1-t0)*1000)}ms | Confidence: {confidence}")
            print(f"Model Bug Identified:\n  {response_data.get('bug_identified')}")
            print(f"Model Root Cause:\n  {response_data.get('root_cause')}")
            print(f"Model Fix Preview:\n  {fixed_code.splitlines()[0] if fixed_code else 'N/A'} ...")
            
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*65)
    print(f"🏁 Final Hard Benchmark Score: {passed} / {len(HARD_TEST_CASES)} passed ({warnings} warnings)")
    print("="*65)

if __name__ == '__main__':
    run_tests()
