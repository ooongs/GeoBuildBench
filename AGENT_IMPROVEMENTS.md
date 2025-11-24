# Agent 히스토리 추적 개선

## 📝 개요

ReAct Agent가 이전 시도들로부터 **적극적으로 학습**할 수 있도록 히스토리 추적 및 분석 기능을 대폭 개선했습니다.

## 🔴 이전 문제점

### 1. 제한적인 히스토리
```python
# 이전 코드
f"Thought: {step.thought.content[:100]}..."  # 100자만!
f"Error: {step.observation.error[:100]}..."  # 100자만!
```
- Thought 내용: **100자로 잘림**
- 에러 메시지: **100자로 잘림**
- DSL 코드: **히스토리에 포함되지 않음**

### 2. 단일 이미지만 제공
```python
# 이전 코드
last_image = memory.steps[-1].observation.image_base64  # 마지막 1개만
```
- 이전 시도들과 비교 불가
- 진행 상황 파악 어려움

### 3. 분석 기능 부재
- 실패 패턴 분석 없음
- 반복되는 실수 감지 못함
- 진행 추세 파악 불가

## ✅ 개선 사항

### 1. 전체 히스토리 제공

#### Before (100자 제한)
```
Iteration 1:
Thought: I need to create three points A, B, C to form a triangle. However, I also need to...
Action: generate_dsl
Result: ✗ Failed
Error: KeyError: 'polygon'
Full traceback:
  File test.py line 10
    polygon...
```

#### After (전체 내용)
```
**Iteration 1: ✗ Failed**

**Thought:**
I need to create three points A, B, C to form a triangle. However, I also need to
ensure they are not collinear, so I'll define them as random points and then create
a polygon from them.

**Action:** generate_dsl

**DSL Code:**
```
point :  -> A
point :  -> B
point :  -> C
polygon : A B C -> tri c a b
equality : A A -> expr0
prove : expr0 -> result
```

**Observation:**
- Execution failed

**Error Details:**
```
KeyError: 'polygon'
Full traceback:
  File "/Users/test/random_constr.py", line 110, in apply
    f = command_dict[name]
        ~~~~~~~~~~~~^^^^^^
KeyError: 'polygon'
```
```

### 2. 여러 이미지 제공

```python
# 개선된 코드
recent_images = []
max_images = 3
for step in reversed(memory.steps[-max_images:]):
    if step.observation.has_image:
        recent_images.append({
            'iteration': step.iteration,
            'success': step.observation.success,
            'image': step.observation.image_base64
        })
```

**효과:**
- 최근 3개 iteration의 이미지를 모두 볼 수 있음
- 진행 상황을 시각적으로 비교 가능
- 어떤 변경이 개선/악화를 가져왔는지 파악 가능

### 3. 실패 패턴 분석

#### 새로운 기능: `get_failure_analysis()`

```python
analysis = memory.get_failure_analysis()

# 결과 예시:
{
    "total_failures": 4,
    "failure_rate": 0.8,
    "common_errors": {
        "KeyError: 'line'": 2,
        "KeyError: 'rotate'": 1,
        "Syntax error": 1
    },
    "repeated_mistakes": [
        {"error": "KeyError: 'line'", "count": 2}
    ]
}
```

**Agent에게 제공되는 정보:**
- 가장 흔한 에러 Top 5
- 2번 이상 반복된 실수
- 전체 실패율

### 4. 진행 상황 요약

#### 새로운 기능: `get_progress_summary()`

```python
summary = memory.get_progress_summary()
```

**출력 예시:**
```
**Progress Summary (Total: 5 iterations)**
- Successful executions: 1
- Failed executions: 4
- Recent trend: ⚡ Mixed results - getting closer

**Most Common Errors:**
  - KeyError: 'line' (occurred 2x)
  - KeyError: 'rotate' (occurred 1x)
  - Syntax error (occurred 1x)

**⚠️ Repeated Mistakes - Avoid These:**
  - KeyError: 'line' (failed 2x)
```

**추세 분석:**
- ✓ 최근 모두 성공: "All recent attempts successful!"
- ✗ 최근 모두 실패: "All recent attempts failed - try a different approach"
- ⚡ 혼합 결과: "Mixed results - getting closer"

### 5. 개선된 프롬프트

#### 프롬프트에 추가된 내용

```markdown
## Learning from History:

**IMPORTANT**: Review the history above carefully:
- Look at which approaches succeeded and which failed
- Identify the root cause of errors (syntax, logic, missing objects)
- If you made the same mistake multiple times, try a different strategy
- If syntax errors occurred, double-check DSL syntax in this attempt
- Compare rendered images to see if you're making progress
```

## 📊 비교 표

| 기능 | 이전 | 개선 후 |
|------|------|--------|
| Thought 내용 | 100자 | **전체** |
| 에러 메시지 | 100자 | **전체** |
| DSL 코드 | ❌ 없음 | ✅ **전체 포함** |
| 이미지 | 1개 (마지막) | **3개 (최근)** |
| 실패 분석 | ❌ 없음 | ✅ **패턴 분석** |
| 반복 실수 감지 | ❌ 없음 | ✅ **자동 감지** |
| 진행 추세 | ❌ 없음 | ✅ **자동 분석** |

## 🎯 실제 효과

### Before: Agent의 제한적인 컨텍스트
```
Agent: "I got an error... KeyError: 'line'..."
       (100자만 보임, 전체 에러 모름)
       (이전 DSL 코드 안 보임)
       (왜 실패했는지 정확히 모름)
       
→ 같은 실수 반복 가능성 높음
```

### After: Agent의 풍부한 컨텍스트
```
Agent: "I see the full error: KeyError: 'line' at line 110
        The DSL code I wrote was: line : A B -> line_AB
        This is the 2nd time I made this mistake
        I need to define A and B BEFORE using them in line
        Looking at the 3 images, I can see progress but still wrong
        
→ 같은 실수 반복하지 않고 새로운 접근
```

## 🔧 기술적 세부사항

### 파일 수정 사항

#### 1. `react_agent.py`
- `_format_history()`: 전체 내용 포함, DSL 코드 추가, 구조화된 포맷
- `_react_step()`: 여러 이미지 제공, 진행 요약 추가

#### 2. `agent_memory.py`
- `get_failure_analysis()`: 실패 패턴 분석
- `get_progress_summary()`: 진행 상황 및 추세 분석

#### 3. `prompts/react_template.txt`
- 학습 원칙 추가
- 히스토리 활용 지침 추가

## 📈 기대 효과

1. **더 빠른 수렴**: 같은 실수 반복하지 않음
2. **더 나은 디버깅**: 전체 에러 메시지로 정확한 원인 파악
3. **패턴 학습**: 반복되는 실수 자동 감지 및 회피
4. **시각적 피드백**: 여러 이미지 비교로 진행 상황 파악
5. **적응적 전략**: 추세 분석으로 접근 방법 조정

## 🧪 테스트

`test_agent_improvements.py`에서 모든 기능 검증 완료:

```bash
python test_agent_improvements.py
```

**결과:**
```
✅ Full thought content (not truncated to 100 chars)
✅ Complete DSL code in history
✅ Full error messages (not truncated to 100 chars)
✅ Multiple images (last 3 iterations)
✅ Failure pattern analysis
✅ Common error detection
✅ Repeated mistake tracking
✅ Progress summary for agent
✅ Trend analysis (improving/declining)
```

## 🎉 결론

이제 Agent는:
- ✅ **과거를 기억**하고
- ✅ **실수로부터 학습**하며
- ✅ **패턴을 인식**하고
- ✅ **적응적으로 전략을 수정**합니다!

이전보다 훨씬 더 **지능적**이고 **효과적**인 문제 해결이 가능합니다.


