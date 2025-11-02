



المبدأ الأساسي

بدلاً من تمثيل الحالة كقيمة مطلقة `s = current_value`، يستخدم النظام تمثيلاً علائقياً:

```
HierarchicalState = {
    progress_ratio: current / target,
    remaining_ratio: (target - current) / target,
    time_ratio: step / max_steps,
    log_gap: log(gap + 1) / log(target + 1),
    gap_magnitude: gap / target,
    constraint_features: f(current, forbidden_states),
    phase_indicator: classify_phase(gap, target),
    efficiency_metrics: compute_efficiency(current, target, step)
}
```

#### 1.2 الميزات العلائقية الأساسية

**أ) نسبة التقدم (Progress Ratio)**
```
progress_ratio = current_position / target_position
```
- قيمة في النطاق [0, ∞)
- مستقلة عن حجم المسألة المطلق
- تمكن المقارنة عبر أهداف مختلفة

**ب) تحليل الفجوة متعدد المقاييس**
```
linear_gap = |target - current| / target
logarithmic_gap = log(|target - current| + 1) / log(target + 1)
```
- التمثيل الخطي للقرارات المحلية
- التمثيل اللوغاريتمي للتخطيط الاستراتيجي

**ج) ميزات القيود**
```
danger_proximity = min(|current - forbidden_i| for forbidden_i in forbidden_set)
constraint_pressure = count_blocked_path_states / total_path_length
```

#### 1.3 خوارزمية حساب الميزات

```python
def compute_relational_features(current, target, step, max_steps, forbidden_states):
    """
    حساب الميزات العلائقية للحالة الحالية
    """
    # التحقق من صحة المدخلات
    if target == 0:
        return zero_vector(feature_dimension)
    
    # الميزات الأساسية
    progress_ratio = current / target
    remaining_ratio = (target - current) / target
    time_ratio = step / max_steps
    
    # تحليل الفجوة
    gap = abs(target - current)
    log_gap = math.log(gap + 1) / math.log(target + 1)
    gap_magnitude = gap / target
    
    # مؤشرات استراتيجية
    is_close = 1.0 if gap <= proximity_threshold else 0.0
    is_far = 1.0 if gap >= target * distance_threshold else 0.0
    
    # ميزات القيود
    danger_proximity = compute_constraint_proximity(current, forbidden_states)
    constraint_pressure = compute_path_blockage(current, target, forbidden_states)
    
    # تحديد المرحلة
    phase = identify_problem_phase(gap, target)
    
    # مقاييس الكفاءة
    theoretical_min = ceil(gap / max_action_size)
    efficiency_ratio = theoretical_min / (max_steps - step + 1)
    
    return [progress_ratio, remaining_ratio, time_ratio, log_gap, gap_magnitude,
            is_close, is_far, danger_proximity, constraint_pressure, phase,
            theoretical_min, efficiency_ratio]
```

### 2. التحليل الهرمي للأهداف

#### 2.1 خوارزمية تحليل الأهداف

```python
def decompose_target_hierarchically(current_position, final_target):
    """
    تحليل الهدف النهائي إلى سلسلة من الأهداف الفرعية القابلة للإدارة
    """
    gap = abs(final_target - current_position)
    
    # للأهداف الصغيرة: نهج مباشر
    if gap <= small_target_threshold:
        return [final_target]
    
    # للأهداف الكبيرة: إنشاء نقاط وسطية استراتيجية
    direction = 1 if final_target > current_position else -1
    num_subgoals = max(min_subgoals, gap // subgoal_spacing)
    step_size = gap // num_subgoals
    
    subgoals = []
    for i in range(1, num_subgoals):
        subgoal = current_position + (step_size * i * direction)
        # تجنب الحالات المحظورة في الأهداف الفرعية
        adjusted_subgoal = adjust_for_constraints(subgoal, forbidden_states)
        subgoals.append(adjusted_subgoal)
    
    # إضافة الهدف النهائي
    subgoals.append(final_target)
    
    return subgoals
```

#### 2.2 إدارة الأهداف الفرعية

```python
class SubgoalManager:
    """
    إدارة تسلسل الأهداف الفرعية وتتبع التقدم
    """
    def __init__(self):
        self.subgoal_stack = []
        self.current_subgoal = None
        self.completed_subgoals = []
    
    def update_subgoals(self, current_position, final_target):
        """تحديث قائمة الأهداف الفرعية عند الحاجة"""
        if not self.subgoal_stack:
            self.subgoal_stack = decompose_target_hierarchically(current_position, final_target)
            self.current_subgoal = self.subgoal_stack.pop(0)
    
    def check_subgoal_completion(self, current_position):
        """فحص اكتمال الهدف الفرعي الحالي"""
        if current_position == self.current_subgoal:
            self.completed_subgoals.append(self.current_subgoal)
            if self.subgoal_stack:
                self.current_subgoal = self.subgoal_stack.pop(0)
                return True  # انتقال إلى هدف فرعي جديد
        return False  # لا يزال يعمل على الهدف الحالي
```

### 3. بنية الشبكة العصبية التكيفية

#### 3.1 التصميم المعماري

```python
class HierarchicalQNetwork(nn.Module):
    """
    شبكة عصبية متعددة الرؤوس مع تخصص للمراحل
    """
    def __init__(self, state_dimension=12, action_dimension=3, hidden_dimension=256):
        super().__init__()
        
        # مستخرج الميزات المشترك
        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(state_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # رؤوس متخصصة للمراحل المختلفة
        self.exploration_head = self._build_specialized_head(hidden_dimension, action_dimension)
        self.navigation_head = self._build_specialized_head(hidden_dimension, action_dimension)
        self.precision_head = self._build_specialized_head(hidden_dimension, action_dimension)
        
        # مصنف المراحل
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dimension, phase_classifier_hidden),
            nn.ReLU(),
            nn.Linear(phase_classifier_hidden, num_phases),
            nn.Softmax(dim=-1)
        )
    
    def _build_specialized_head(self, input_dim, output_dim):
        """بناء رأس متخصص لمرحلة معينة"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, state_features):
        """التمرير الأمامي مع دمج المراحل"""
        # استخراج الميزات المشتركة
        shared_features = self.shared_feature_extractor(state_features)
        
        # تصنيف المرحلة الحالية
        phase_probabilities = self.phase_classifier(shared_features)
        
        # حساب Q-values لكل مرحلة
        exploration_q = self.exploration_head(shared_features)
        navigation_q = self.navigation_head(shared_features)
        precision_q = self.precision_head(shared_features)
        
        # الدمج المرجح بناءً على احتماليات المراحل
        combined_q_values = (
            phase_probabilities[:, 0:1] * exploration_q +
            phase_probabilities[:, 1:2] * navigation_q +
            phase_probabilities[:, 2:3] * precision_q
        )
        
        return combined_q_values, phase_probabilities
```

#### 3.2 تصنيف المراحل

```python
def identify_problem_phase(gap, target):
    """
    تحديد مرحلة حل المسألة الحالية
    المراحل: 0=استكشاف، 1=تنقل، 2=دقة
    """
    gap_ratio = gap / target
    
    if gap_ratio > exploration_threshold:
        return 0.0  # مرحلة الاستكشاف - حركات واسعة
    elif gap > precision_threshold:
        return 1.0  # مرحلة التنقل - حركات متوسطة مع تجنب القيود
    else:
        return 2.0  # مرحلة الدقة - حركات صغيرة للوصول الدقيق
```

### 4. خوارزمية التدريب

#### 4.1 التعلم المنهجي (Curriculum Learning)

```python
def curriculum_training_schedule():
    """
    جدولة التدريب المنهجي من البسيط إلى المعقد
    """
    return [
        {
            'stage_name': 'foundation',
            'target_range': range(5, 11),
            'episodes': 1000,
            'max_steps': 20,
            'constraint_complexity': 'low'
        },
        {
            'stage_name': 'expansion', 
            'target_range': range(12, 31),
            'episodes': 1500,
            'max_steps': 25,
            'constraint_complexity': 'medium'
        },
        {
            'stage_name': 'generalization',
            'target_range': range(20, 101),
            'episodes': 1500,
            'max_steps': 30,
            'constraint_complexity': 'high'
        },
        {
            'stage_name': 'mixed_training',
            'target_range': range(5, 101),
            'episodes': 1000,
            'max_steps': 30,
            'constraint_complexity': 'variable'
        }
    ]
```

#### 4.2 دالة المكافآت المطورة

```python
def compute_enhanced_reward(current_state, next_state, target, step, forbidden_states):
    """
    حساب المكافأة مع تشكيل متطور للسلوك المرغوب
    """
    # مكافأة النجاح مع مكافأة الوقت
    if next_state == target:
        time_bonus = max(0, max_steps - step)
        return success_reward + time_bonus
    
    # عقوبة انتهاك القيود
    if next_state in forbidden_states:
        return constraint_violation_penalty
    
    # مكافأة التقدم
    current_distance = abs(target - current_state)
    next_distance = abs(target - next_state)
    if next_distance < current_distance:
        progress_reward = base_progress_reward * (current_distance - next_distance)
        return progress_reward
    
    # عقوبة تجاوز الهدف
    if next_state > target and current_state <= target:
        return overshoot_penalty
    
    # عقوبة الوقت الأساسية
    return time_penalty
```

### 5. خوارزمية اختيار الحركة مع مراعاة القيود

```python
def constraint_aware_action_selection(agent, current_state, target, step, max_steps, forbidden_states):
    """
    اختيار الحركة مع مراعاة القيود والتفكير الهرمي
    """
    # تحديث الأهداف الفرعية
    agent.subgoal_manager.update_subgoals(current_state, target)
    working_target = agent.subgoal_manager.current_subgoal or target
    
    # حساب الميزات العلائقية
    state_features = compute_relational_features(current_state, working_target, 
                                               step, max_steps, forbidden_states)
    
    # الحصول على Q-values من الشبكة
    with torch.no_grad():
        q_values, phase_probs = agent.network(torch.FloatTensor(state_features).unsqueeze(0))
    
    # تصفية الحركات غير الصالحة
    valid_actions = []
    for action_idx, action in enumerate(agent.action_space):
        next_state = current_state + action
        
        # فحص انتهاك القيود
        if forbidden_states and next_state in forbidden_states:
            continue
        
        # فحص الحدود الفيزيائية
        if next_state < 0:  # مثال: لا يمكن أن تكون القيمة سالبة
            continue
            
        valid_actions.append((action_idx, action))
    
    # التعامل مع حالة عدم وجود حركات صالحة
    if not valid_actions:
        return emergency_backtrack_action
    
    # اختيار أفضل حركة صالحة
    best_action_idx = -1
    best_q_value = float('-inf')
    
    for action_idx, action in valid_actions:
        if q_values[0, action_idx] > best_q_value:
            best_q_value = q_values[0, action_idx]
            best_action_idx = action_idx
    
    return agent.action_space[best_action_idx]
```

---










