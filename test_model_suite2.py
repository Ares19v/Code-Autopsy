import urllib.request
import json
import time

API_URL = "http://localhost:8000/review"

TEST_CASES_SET_2 = [
    {
        "name": "1. Python Modifying List While Iterating",
        "language": "python",
        "code": "def remove_negatives(numbers):\n    for num in numbers:\n        if num < 0:\n            numbers.remove(num)\n    return numbers",
        "expected_bug_keywords": ["iterat", "modif", "remove", "index", "skip"]
    },
    {
        "name": "2. JS Alphabetical Array.sort() on Numbers",
        "language": "javascript",
        "code": "function sortNumbers(arr) {\n    return arr.sort();\n}",
        "expected_bug_keywords": ["sort", "string", "lexicographical", "compare", "alphabetical", "comparator", "(a, b)"]
    },
    {
        "name": "3. Python UnboundLocalError (Global Counter)",
        "language": "python",
        "code": "counter = 0\ndef increment():\n    counter += 1\n    return counter",
        "expected_bug_keywords": ["global", "unboundlocalerror", "scope", "local", "assignment"]
    },
    {
        "name": "4. JS NaN Equality Comparison (val === NaN)",
        "language": "javascript",
        "code": "function isNotANumber(val) {\n    return val === NaN;\n}",
        "expected_bug_keywords": ["nan", "isnan", "number.isnan", "equal"]
    },
    {
        "name": "5. Python Nested Object Shallow Copy Mutation",
        "language": "python",
        "code": "def clone_and_modify(matrix):\n    new_matrix = matrix.copy()\n    new_matrix[0][0] = 999\n    return new_matrix",
        "expected_bug_keywords": ["shallow", "deep", "deepcopy", "nested", "mutat", "reference"]
    },
    {
        "name": "6. JS forEach with Async/Await (Unawaited Promises)",
        "language": "javascript",
        "code": "async function sendAll(users, sendEmail) {\n    users.forEach(async (user) => {\n        await sendEmail(user);\n    });\n    console.log('All sent');\n}",
        "expected_bug_keywords": ["foreach", "promise.all", "for...of", "await", "async", "concurrent"]
    },
    {
        "name": "7. Python NoneType Attribute Access (.get chain)",
        "language": "python",
        "code": "def get_user_email(user_data):\n    return user_data.get('profile').get('email')",
        "expected_bug_keywords": ["none", "nonetype", "attribute", "error", "profile", "null"]
    },
    {
        "name": "8. JS Floating Point Precision Equality (0.1 + 0.2 === 0.3)",
        "language": "javascript",
        "code": "function checkTotal(priceA, priceB, expected) {\n    return priceA + priceB === expected;\n}",
        "expected_bug_keywords": ["float", "precision", "rounding", "epsilon", "0.1", "binary"]
    },
    {
        "name": "9. Python SQL Injection via f-string formatting",
        "language": "python",
        "code": "def get_user(cursor, username):\n    query = f\"SELECT * FROM users WHERE username = '{username}'\"\n    return cursor.execute(query)",
        "expected_bug_keywords": ["sql", "injection", "parameter", "format", "f-string", "sanitize", "vulnerability"]
    },
    {
        "name": "10. JS Loss of 'this' Context in Callback",
        "language": "javascript",
        "code": "class Timer {\n    constructor() {\n        this.seconds = 0;\n    }\n    start() {\n        setInterval(function() {\n            this.seconds++;\n        }, 1000);\n    }\n}",
        "expected_bug_keywords": ["this", "arrow", "bind", "scope", "context", "callback"]
    }
]

def run_tests():
    print("="*65)
    print("🔬 Code Autopsy - Benchmark Suite 2 (10 New Problem Types)")
    print("="*65)
    
    passed = 0
    warnings = 0
    
    for i, test in enumerate(TEST_CASES_SET_2):
        print(f"\n─────────────────────────────────────────────────────────────")
        print(f"▶ Test {i+1}: {test['name']} ({test['language']})")
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
                    status = "⚠️ WARNING (Identified a bug, but matched keyword set differs)"
                    warnings += 1
            
            print(f"Status: {status}")
            print(f"Latency: {round((t1-t0)*1000)}ms | Confidence: {confidence}")
            print(f"Model Bug Identified:\n  {response_data.get('bug_identified')}")
            print(f"Model Root Cause:\n  {response_data.get('root_cause')}")
            print(f"Model Fix Preview:\n  {fixed_code.splitlines()[0] if fixed_code else 'N/A'} ...")
            
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*65)
    print(f"🏁 Final Benchmark Suite 2 Score: {passed} / {len(TEST_CASES_SET_2)} passed ({warnings} warnings)")
    print("="*65)

if __name__ == '__main__':
    run_tests()
