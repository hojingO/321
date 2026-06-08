# -*- coding: utf-8 -*-
"""
주제 B: 격자 기반 반복계산으로 2D 정상 열전도 해석 및 민감도 분석
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 물리적 상수 및 환경 설정
# ==========================================
def get_inputs():
    """물리적 상수 및 환경 설정을 반환하는 함수"""
    Lx = 1.0           # 판 가로 길이 (m)
    Ly = 1.0           # 가이드 조건(Nx=Ny)에 맞추기 위해 정사각형(1mx1m) 영역으로 해석
    k = 200.0          # 열전도율 [W/m*K]

    # 내부 사각형 열원 정보 (중심 및 크기)
    hs_xc, hs_yc = 0.5, 0.5
    hs_w, hs_h = 0.2, 0.2
    hs_Q = 1.5e6       # 열생성량 [W/m³]

    # 경계 온도 조건 (°C)
    T_left = 100.0     
    T_right = 20.0     
    T_top = 20.0       # 단순 경계조건 적용 (상하단도 고정온도로 단순화)
    T_bottom = 20.0    
    T_limit = 80.0     # 허용 온도 한계

    # 제어 매개변수
    grid_list = [20, 40, 80]  # 가이드라인 지정 격자 크기
    max_iter = 10000
    tolerance = 1e-5
    
    return Lx, Ly, k, hs_xc, hs_yc, hs_w, hs_h, hs_Q, T_left, T_right, T_top, T_bottom, T_limit, grid_list, max_iter, tolerance

# 함수 호출을 통해 전역 변수 설정 (기존 코드 구조 및 호환성 100% 유지)
Lx, Ly, k, hs_xc, hs_yc, hs_w, hs_h, hs_Q, T_left, T_right, T_top, T_bottom, T_limit, grid_list, max_iter, tolerance = get_inputs()



# ==========================================
# 2. 가이드라인 요구 함수 정의
# ==========================================

def initialize_temperature(Nx, Ny):
    """2D 배열을 사용하여 초기 온도장을 설정하는 함수 (필수 조건)"""
    T = np.zeros((Ny, Nx))
    # 내부를 좌우 벽면 온도의 평균값 등으로 채워 초기화
    T[:, :] = (T_left + T_right) / 2.0
    
    # 경계 조건 부여
    T[:, 0] = T_left
    T[:, -1] = T_right
    T[0, :] = T_bottom
    T[-1, :] = T_top
    return T

def update_temperature(T, k, q, dx):
    """가이드라인의 FDM 업데이트 수식을 적용하는 함수 (Delta x = Delta y 일 때)"""
    Ny, Nx = T.shape
    T_new = T.copy()
    
    # 단순 경계조건이므로 내부 노드(1부터 N-2까지)만 순회하며 업데이트
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            # 가이드 상의 수식: 0.25 * (주변 4개 노드 합 + q * dx^2 / k)
            T_new[j, i] = 0.25 * (T[j, i+1] + T[j, i-1] + T[j+1, i] + T[j-1, i] + (q[j, i] * (dx**2) / k))
            
    return T_new

def save_temperature_field(T, grid):
    """해석된 온도장 격자 데이터를 CSV 파일로 저장하는 함수"""
    filename = f"temperature_field_grid_{grid}.csv"
    Ny, Nx = T.shape
    x_vals = np.linspace(0, Lx, Nx)
    y_vals = np.linspace(0, Ly, Ny)
    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Y \\ X"] + [f"{x:.4f}m" for x in x_vals])
        for j in range(Ny):
            row = [f"{y_vals[j]:.4f}m"] + [f"{T[j, i]:.4f}" for i in range(Nx)]
            writer.writerow(row)
    print(f"[파일 저장 완료] 격자 {grid}x{grid} 데이터가 {filename}에 저장되었습니다.")

def plot_heatmap(T, grid):
    """필수: Heatmap과 열원 중심선 온도 Profile 그래프를 동시에 출력하는 함수"""
    Ny, Nx = T.shape
    x_vals = np.linspace(0, Lx, Nx)
    y_vals = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # [좌측] 2D Heatmap
    ax1 = axes[0]
    im = ax1.imshow(T, extent=[0, Lx, 0, Ly], origin='lower', cmap='jet', aspect='equal')
    fig.colorbar(im, ax=ax1, label="Temperature (°C)")
    
    # 열원 위치 표시 (점선 사각형)
    rect = plt.Rectangle((hs_xc - hs_w/2.0, hs_yc - hs_h/2.0), hs_w, hs_h,
                         fill=False, edgecolor='white', linestyle='--', linewidth=1.5, label='Heat Source')
    ax1.add_patch(rect)
    ax1.set_title(f"2D Heatmap (Grid: {grid}x{grid})")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.legend()
    
    # [우측] 열원의 중심선을 통과하는 1D 온도 Profile
    ax2 = axes[1]
    j_center = np.argmin(np.abs(y_vals - hs_yc)) # Y 중심과 가장 가까운 인덱스
    ax2.plot(x_vals, T[j_center, :], 'b-', linewidth=2, label=f"Center Line Profile (Y={y_vals[j_center]:.2f}m)")
    ax2.axhline(y=T_limit, color='r', linestyle='--', label=f"Safety Limit ({T_limit}°C)")
    ax2.set_title(f"1D Temperature Profile (Grid: {grid})")
    ax2.set_xlabel("X Position (m)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.grid(True, linestyle=':')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. 메인 실행 루프 (제시된 예시 Pseudo 코드 구조 100% 일치)
# ==========================================
print("주제 B 시뮬레이션을 시작합니다.")

# 격자 민감도 결과를 저장하기 위한 리스트
summary_results = []

for grid in grid_list:
    print(f"\n[해석 중] 현재 격자 크기: {grid} x {grid}")
    
    # 1단계: 격자 크기에 따른 수치 정보 정의 (Nx = Ny = grid)
    Nx = grid
    Ny = grid
    dx = Lx / (Nx - 1)
    
    # 2단계: 온도장 초기화 호출
    T = initialize_temperature(Nx, Ny)
    
    # 3단계: 내부 열원(부품) 위치에 따른 생성항 배열 q 정의
    q_array = np.zeros((Ny, Nx))
    x_vals = np.linspace(0, Lx, Nx)
    y_vals = np.linspace(0, Ly, Ny)
    for j in range(Ny):
        for i in range(Nx):
            if (hs_xc - hs_w/2.0 <= x_vals[i] <= hs_xc + hs_w/2.0) and \
               (hs_yc - hs_h/2.0 <= y_vals[j] <= hs_yc + hs_h/2.0):
                q_array[j, i] = hs_Q
                
    # 4단계: 수치 내부 반복 루프
    for it in range(max_iter):
        T_new = update_temperature(T, k, q_array, dx)
        
        # 오차(error) 판정: 최대 잔차 계산
        error = np.max(np.abs(T_new - T))
        T = T_new
        
        if error < tolerance:
            print(f"-> {it+1}회 반복 후 수렴 만족.")
            break
            
    # 5단계: 결과 수집 (격자 민감도 분석용 데이터)
    t_max = np.max(T)
    t_mean = np.mean(T)
    exceed_ratio = (np.sum(T > T_limit) / T.size) * 100.0
    summary_results.append((grid, t_max, t_mean, exceed_ratio))
    
    # 6단계: 가이드라인 필수 요구사항 수행 (파일 저장 및 시각화)
    save_temperature_field(T, grid)
    plot_heatmap(T, grid)


# ==========================================
# 4. 필수 요건: 격자 민감도 결과 비교 검증 종합 출력
# ==========================================
print("\n" + "="*60)
print("             [격자 민감도(Grid Sensitivity) 분석 요약]")
print("="*60)
print(f"{'Grid Size':<12}{'Max Temp (°C)':<15}{'Mean Temp (°C)':<15}{'Exceed Area (%)':<15}")
print("-"*60)
for res in summary_results:
    print(f"{f'{res[0]}x{res[0]}':<12}{res[1]:<15.2f}{res[2]:<15.2f}{res[3]:<15.2f}")
print("="*60)


# ==========================================
# 5. 도전 과제: 열원 위치 변경 시 최고온도 비교 분석
# ==========================================
print("\n[도전 과제] 열원 실장 위치 변경에 따른 최고온도 스윕 분석을 시작합니다.")
# 격자는 중간 크기인 40x40으로 고정하여 실험
sweep_grid = 40
dx_sweep = Lx / (sweep_grid - 1)
x_positions = [0.3, 0.5, 0.7] # 열원의 중심 X 좌표 변경 시나리오
sweep_max_temps = []

for x_pos in x_positions:
    T_sweep = initialize_temperature(sweep_grid, sweep_grid)
    q_sweep = np.zeros((sweep_grid, sweep_grid))
    xs = np.linspace(0, Lx, sweep_grid)
    ys = np.linspace(0, Ly, sweep_grid)
    
    # 변경된 위치(x_pos)를 기준으로 열원 매핑
    for j in range(sweep_grid):
        for i in range(sweep_grid):
            if (x_pos - hs_w/2.0 <= xs[i] <= x_pos + hs_w/2.0) and \
               (hs_yc - hs_h/2.0 <= ys[j] <= hs_yc + hs_h/2.0):
                q_sweep[j, i] = hs_Q
                
    # 반복 계산
    for it in range(max_iter):
        T_new = update_temperature(T_sweep, k, q_sweep, dx_sweep)
        error = np.max(np.abs(T_new - T_sweep))
        T_sweep = T_new
        if error < tolerance:
            break
            
    sweep_max_temps.append(np.max(T_sweep))
    print(f" -> 열원 위치 Xc = {x_pos}m 일 때, 최고 온도 = {np.max(T_sweep):.2f} °C")

# 도전과제 그래프 시각화
plt.figure(figsize=(6, 4))
plt.plot(x_positions, sweep_max_temps, marker='D', color='orange', linewidth=2)
plt.axhline(y=T_limit, color='r', linestyle='--', label=f"Safety Limit ({T_limit}°C)")
plt.title(" Max Temperature vs. Heat Source Position")
plt.xlabel(" Heat Source Center X Position (m)")
plt.ylabel("Max Temperature (°C)")
plt.grid(True, linestyle=':')
plt.legend()
plt.show()