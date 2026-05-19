"""极简专家系统 Demo — 第 2 讲：经典 AI 的黄金年代

展示基于 if-then 规则的专家系统的优雅与脆弱性。
--add-conflict 标志加入冲突规则，展示规则系统的维护地狱。

用法：
    python expert_system_mini.py                # 基本规则集
    python expert_system_mini.py --add-conflict # 加入冲突规则
    python expert_system_mini.py --verbose      # 显示推理过程
"""

import argparse


# ---------- Rule Engine ----------

class Rule:
    """A single IF-THEN rule with confidence."""
    def __init__(self, conditions: set, conclusion: str, confidence: float, source: str = ""):
        self.conditions = conditions
        self.conclusion = conclusion
        self.confidence = confidence
        self.source = source

    def matches(self, facts: set) -> bool:
        return self.conditions.issubset(facts)

    def __repr__(self):
        conds = " AND ".join(sorted(self.conditions))
        return f"IF {conds} THEN {self.conclusion} (conf={self.confidence:.1f}) [{self.source}]"


class ExpertSystem:
    """A minimal forward-chaining expert system."""
    def __init__(self):
        self.rules: list[Rule] = []

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def diagnose(self, symptoms: set, verbose: bool = False) -> list[tuple[str, float, str]]:
        """Forward chaining: match all applicable rules."""
        results = []
        for rule in self.rules:
            if rule.matches(symptoms):
                results.append((rule.conclusion, rule.confidence, str(rule)))
                if verbose:
                    print(f"  [FIRED] {rule}")
        return results


# ---------- Knowledge Base ----------

def build_base_rules() -> list[Rule]:
    """基础规则集：简单、无冲突。"""
    return [
        Rule({"fever", "cough"},
             "Common Cold", 0.6, "Rule-1: 感冒"),
        Rule({"fever", "cough", "chest_pain"},
             "Pneumonia", 0.7, "Rule-2: 肺炎"),
        Rule({"fever", "headache", "stiff_neck"},
             "Meningitis", 0.8, "Rule-3: 脑膜炎"),
        Rule({"sneeze", "runny_nose"},
             "Allergy", 0.5, "Rule-4: 过敏"),
        Rule({"fever", "rash"},
             "Measles", 0.7, "Rule-5: 麻疹"),
    ]


def build_conflict_rules() -> list[Rule]:
    """冲突规则：新规则与旧规则矛盾。"""
    return [
        Rule({"fever", "cough", "headache"},
             "Meningitis", 0.6, "Rule-6: 新增·脑膜炎变体"),
        # ^ Conflicts with Rule-1 (fever+cough→Cold) when headache is also present
        Rule({"fever", "cough", "headache"},
             "Severe Flu", 0.7, "Rule-7: 新增·重流感"),
        # ^ Conflicts with Rule-6 on the same conditions!
        Rule({"fever", "rash", "cough"},
             "COVID-19", 0.5, "Rule-8: 新增·新冠"),
        # ^ Conflicts with Rule-5 (fever+rash→Measles) and Rule-1
    ]


# ---------- Demo Scenarios ----------

def run_demo(es: ExpertSystem, verbose: bool):
    scenarios = [
        ("Patient A", {"fever", "cough"}),
        ("Patient B", {"fever", "headache", "stiff_neck"}),
        ("Patient C", {"fever", "cough", "headache"}),
        ("Patient D", {"fever", "rash", "cough"}),
        ("Patient E", {"sneeze", "runny_nose"}),
    ]

    for name, symptoms in scenarios:
        print(f"\n{'─'*50}")
        print(f"  {name}: {', '.join(sorted(symptoms))}")
        print(f"{'─'*50}")
        results = es.diagnose(symptoms, verbose=verbose)
        if not results:
            print("  → No matching diagnosis.")
        elif len(results) == 1:
            diag, conf, _ = results[0]
            print(f"  → Diagnosis: {diag} (confidence: {conf:.0%})")
        else:
            print(f"  → ⚠️  CONFLICT: {len(results)} rules fired!")
            for diag, conf, rule_str in results:
                print(f"     • {diag} (confidence: {conf:.0%})")
            print(f"  → System cannot decide. This is the maintenance hell.")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Minimal Expert System Demo")
    parser.add_argument("--add-conflict", action="store_true",
                        help="Add conflicting rules to demonstrate fragility")
    parser.add_argument("--verbose", action="store_true",
                        help="Show which rules fired")
    args = parser.parse_args()

    es = ExpertSystem()

    # Load base rules
    for rule in build_base_rules():
        es.add_rule(rule)

    print("=" * 50)
    print("  EXPERT SYSTEM DEMO — 极简规则引擎")
    print("=" * 50)
    print(f"\n  Rules loaded: {len(es.rules)}")
    if args.add_conflict:
        conflict_rules = build_conflict_rules()
        for rule in conflict_rules:
            es.add_rule(rule)
        print(f"  + Conflict rules added: {len(conflict_rules)}")
        print(f"  Total rules: {len(es.rules)}")
    print()

    if args.verbose:
        print("  All rules:")
        for r in es.rules:
            print(f"    {r}")

    run_demo(es, args.verbose)

    # Summary
    print(f"\n{'='*50}")
    if args.add_conflict:
        print("  💡 Lesson: Adding 3 new rules caused conflicts in")
        print("     multiple diagnoses. In real expert systems with")
        print("     thousands of rules, this is exponentially worse.")
        print("     This is the 'Knowledge Acquisition Bottleneck'.")
    else:
        print("  💡 Clean rules → clean diagnoses.")
        print("     Now try: python expert_system_mini.py --add-conflict")
        print("     to see what happens when rules conflict.")
    print("=" * 50)


if __name__ == "__main__":
    main()
