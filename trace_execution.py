#!/usr/bin/env python3
"""
DSL 실행 과정을 단계별로 추적하는 스크립트
"""
import sys
from random_constr import Construction, parse_command, Element, Command, ConstCommand

# 간단한 예제 DSL
dsl_code = """const int 100 -> x
const int 150 -> y
point : x y -> P
const int 50 -> radius
circle : P radius -> c
segment : P P -> dummy
equality : P P -> result
prove : result -> output"""

print("="*60)
print("DSL 실행 과정 추적")
print("="*60)

print("\n📝 입력 DSL:")
print("-"*60)
print(dsl_code)

print("\n" + "="*60)
print("1단계: 텍스트 파싱")
print("="*60)

element_dict = {}
commands = []
const_commands = []

for i, line in enumerate(dsl_code.strip().split('\n'), 1):
    print(f"\n[Line {i}] {line}")
    cmd = parse_command(line, element_dict)
    
    if isinstance(cmd, ConstCommand):
        print(f"  → ConstCommand: {cmd.datatype.__name__}({cmd.value}) -> {cmd.element.label}")
        const_commands.append(cmd)
    elif isinstance(cmd, Command):
        inputs = ', '.join([el.label for el in cmd.input_elements])
        outputs = ', '.join([el.label if el else '_' for el in cmd.output_elements])
        print(f"  → Command: {cmd.name}({inputs}) -> {outputs}")
        commands.append(cmd)

print("\n" + "="*60)
print("2단계: 상수 명령어 즉시 실행")
print("="*60)

for cmd in const_commands:
    cmd.apply()
    print(f"  {cmd.element.label}.data = {cmd.element.data}")

print("\n" + "="*60)
print("3단계: 일반 명령어 실행")
print("="*60)

for i, cmd in enumerate(commands, 1):
    if cmd.name == "prove":
        continue  # prove는 검증용이므로 스킵
        
    print(f"\n[Command {i}] {cmd.name}")
    
    # 입력 데이터 확인
    print("  입력:")
    for el in cmd.input_elements:
        print(f"    {el.label}.data = {el.data} (type: {type(el.data).__name__})")
    
    # 타입 시그니처
    from random_constr import command_types_name
    input_data = [el.data for el in cmd.input_elements]
    func_name = command_types_name(cmd.name, input_data)
    print(f"  → 호출할 함수: {func_name}()")
    
    # 실행
    try:
        cmd.apply()
        print("  출력:")
        for el in cmd.output_elements:
            if el:
                print(f"    {el.label}.data = {el.data}")
    except Exception as e:
        print(f"  ✗ 오류: {e}")

print("\n" + "="*60)
print("4단계: 최종 상태")
print("="*60)

print("\n생성된 모든 Element:")
for label, element in element_dict.items():
    data_type = type(element.data).__name__ if element.data else "None"
    print(f"  {label}: {data_type}")
    if hasattr(element.data, 'a'):  # Point
        print(f"    좌표: {element.data.a}")
    elif hasattr(element.data, 'r'):  # Circle
        print(f"    중심: {element.data.c}, 반지름: {element.data.r}")

print("\n" + "="*60)
print("5단계: 렌더링 (시뮬레이션)")
print("="*60)

print("\n렌더링 가능한 객체:")
for label, element in element_dict.items():
    if element.drawable():
        print(f"  {label} ({type(element.data).__name__})")
        print(f"    → element.data.draw(ax, corners) 호출됨")

print("\n" + "="*60)
print("✓ 완료!")
print("="*60)

print("\n💡 요약:")
print("  1. DSL 텍스트 → Command 객체")
print("  2. 상수는 즉시 실행")
print("  3. 명령어는 순차 실행")
print("  4. 타입에 따라 다른 함수 호출")
print("  5. Element.data에 결과 저장")
print("  6. draw() 메서드로 렌더링")




