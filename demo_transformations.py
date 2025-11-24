#!/usr/bin/env python3
"""
기하학적 변환 데모: 회전, 평행이동 등
"""
import matplotlib.pyplot as plt
from random_constr import Construction

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("기하학적 변환 데모")
print("="*60)

# 회전 데모
print("\n📐 회전(Rotation) 데모")
print("-"*60)

c = Construction(display_size=(400, 400))
c.load('examples/rotation_demo.txt')
c.generate(max_attempts=10)

fig, ax = plt.subplots(figsize=(8, 8))
c.render(ax)
ax.set_title('회전 변환 데모\n(30°, 60°, 90°, 180° 회전)', fontsize=14, fontweight='bold')
plt.savefig('rotation_demo.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ rotation_demo.png 생성 완료!")

print("\n" + "="*60)
print("지원되는 변환 기능:")
print("="*60)
print("""
1. 🔄 회전 (Rotation)
   - rotate : Point Angle Center -> RotatedPoint
   - 점을 특정 각도만큼 회전시킴
   
2. ➡️  평행이동 (Translation)
   - translate : Point Vector -> TranslatedPoint
   - 점을 벡터 방향으로 이동시킴
   
3. 🔍 (확대/축소는 내장 fit_to_window에서 자동 처리)

사용 예제:
---------
# 30도 회전
const AngleSize 0.523599 -> angle30
rotate : A angle30 Center -> B

# 각도를 이용한 회전
angle : P Q R -> angle_pqr
rotate : A angle_pqr Center -> B

# 평행이동
vector : P1 P2 -> vec
translate : A vec -> B
""")

print("\n✅ 회전 데모 이미지가 생성되었습니다!")
print("   rotation_demo.png를 확인하세요.")





