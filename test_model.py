import urllib.request
import json
import time

API_URL = "http://localhost:8000/review"

TEST_CASES = [
    {
        "name": "1. Python Mutable Default Argument",
        "language": "python",
        "code": "def append_item(val, lst=[]):\n    lst.append(val)\n    return lst",
        "expected_bug_keywords": ["mutable", "default", "shared", "lst=[]"]
    },
    {
        "name": "2. Python ZeroDivisionError",
        "language": "python",
        "code": "def get_average(nums):\n    return sum(nums) / len(nums)",
        "expected_bug_keywords": ["zero", "empty", "division", "len"]
    },
    {
        "name": "3. JS Missing Await",
        "language": "javascript",
        "code": "async function fetchUser(response) {\n    const data = response.json();\n    return data.name;\n}",
        "expected_bug_keywords": ["await", "promise", "asynchronous"]
    },
    {
        "name": "4. JS var Closure in Loop",
        "language": "javascript",
        "code": "for (var i = 0; i < 3; i++) {\n    setTimeout(() => console.log(i), 100);\n}",
        "expected_bug_keywords": ["var", "closure", "scope", "let", "3"]
    },
    {
        "name": "5. Python Dict KeyError",
        "language": "python",
        "code": "def get_value(my_dict, key):\n    return my_dict[key]",
        "expected_bug_keywords": ["keyerror", "exist", ".get", "in"]
    },
    {
        "name": "6. JS React State Direct Mutation",
        "language": "javascript",
        "code": "function addItem(items, newItem) {\n    items.push(newItem);\n    return items;\n}",
        "expected_bug_keywords": ["mutate", "push", "pure", "spread", "copy"]
    },
    {
        "name": "7. Python Unclosed File / Resource Leak",
        "language": "python",
        "code": "def write_data(filename, data):\n    f = open(filename, 'w')\n    f.write(data)",
        "expected_bug_keywords": ["close", "leak", "with", "context"]
    },
    {
        "name": "8. Python IndexError (Off-by-one)",
        "language": "python",
        "code": "def get_last_item(arr):\n    return arr[len(arr)]",
        "expected_bug_keywords": ["index", "out of range", "len", "-1"]
    },
    {
        "name": "9. JS Incorrect Equality Coercion",
        "language": "javascript",
        "code": "function isZero(val) {\n    return val == 0;\n}",
        "expected_bug_keywords": ["===", "strict", "coercion", "type"]
    },
    {
        "name": "10. Python Unreachable Code",
        "language": "python",
        "code": "def do_work():\n    return True\n    print('Work finished')",
        "expected_bug_keywords": ["unreachable", "return", "print"]
    }
]

def run_tests():
    print("="*60)
    print("🔬 Code Autopsy - Model Intelligence Test")
    print("="*60)
    
    passed = 0
    
    for i, test in enumerate(TEST_CASES):
        print(f"\nRunning Test {i+1}: {test['name']}")
        
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
            combined_analysis = bug_identified + " " + root_cause
            
            # Check if model hallucinated "No critical bug"
            if "no critical bug" in combined_analysis or "no specific issue" in combined_analysis:
                status = "❌ FAILED (Model claimed 'No bug found')"
            else:
                # Check if it caught the actual bug
                matched_keywords = [kw for kw in test["expected_bug_keywords"] if kw in combined_analysis]
                if matched_keywords:
                    status = f"✅ PASSED (Matched keywords: {matched_keywords})"
                    passed += 1
                else:
                    status = "⚠️ WARNING (Found a bug, but maybe not the right one)"
            
            print(f"Status: {status}")
            print(f"Latency: {round((t1-t0)*1000)}ms")
            print(f"Model Output (Bug): {response_data.get('bug_identified')}")
            print(f"Model Output (Cause): {response_data.get('root_cause')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60)
    print(f"🏁 Final Score: {passed} / {len(TEST_CASES)} passed")
    print("="*60)

if __name__ == '__main__':
    run_tests()
