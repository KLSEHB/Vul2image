import csv
import re
import sys

# C/C++ 关键字和常见类型（避免重命名）
RESERVED = {
    # C/C++ keywords
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double',
    'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register',
    'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef',
    'union', 'unsigned', 'void', 'volatile', 'while',
    # C++ additional
    'bool', 'class', 'const_cast', 'delete', 'dynamic_cast', 'explicit', 'false',
    'friend', 'inline', 'mutable', 'namespace', 'new', 'operator', 'private',
    'protected', 'public', 'reinterpret_cast', 'static_cast', 'template', 'this',
    'throw', 'true', 'try', 'catch', 'typeid', 'typename', 'using', 'virtual', 'wchar_t',
    # common types & macros (optional)
    'size_t', 'NULL', 'nullptr', 'std', 'string', 'vector', 'map', 'cout', 'cin'
}

def anonymize_code(code: str) -> str:
    # 提取所有由字母/下划线开头的标识符（包括数字）
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)

    # 过滤出需要重命名的标识符：不是保留字，且不是纯大写宏（如 MAX_SIZE）
    candidates = set()
    for t in tokens:
        if t not in RESERVED and not (t.isupper() and len(t) > 1):
            candidates.add(t)

    # 构建映射：每个候选名 → id_0, id_1, ...
    name_to_anon = {}
    for i, name in enumerate(sorted(candidates)):  # sorted 保证可复现
        name_to_anon[name] = f'id_{i}'

    # 替换：使用 word boundary \b 避免部分匹配（如 "var" 不替换 "var2" 中的 "var"）
    new_code = code
    for name, anon in sorted(name_to_anon.items(), key=lambda x: -len(x[0])):  # 长名字优先
        # 使用 \b 确保完整单词替换
        new_code = re.sub(r'\b' + re.escape(name) + r'\b', anon, new_code)

    return new_code

def main(input_csv: str, output_csv: str):
    with open(input_csv, 'r', encoding='utf-8') as fin, \
         open(output_csv, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.DictReader(fin)
        fieldnames = ['idx', 'processed_func', 'target']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for row in reader:
            idx = row['idx']
            func_code = row['processed_func']
            target = row['target']

            adv_code = anonymize_code(func_code)

            writer.writerow({
                'idx': idx,
                'processed_func': adv_code,
                'target': target
            })

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} samples...")

    print(f"Done! Saved to {output_csv}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python simple_adversarial.py <input.csv> <output.csv>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])