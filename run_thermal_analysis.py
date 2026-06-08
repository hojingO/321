# -*- coding: utf-8 -*-
"""
2차원 정상 상태 열전도 해석 및 격자 민감도 분석 프로그램 (주제 B)

"""

import math
import csv
import numpy as np
import matplotlib.pyplot as plt

# Standard English fonts will be used for Matplotlib to avoid rendering errors.
# Unicode minus is kept for proper negative sign representation.
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 입력 함수 (get_inputs)
# ==========================================
def get_inputs():
    """
    해석에 필요한 모든 물리적 및 수치적 경계 조건 설정을 사전(dictionary) 형태로 반환하는 함수.

    """
    inputs = {
        # 금속판 크기 설정 (단위: m)
        'Lx': 1.0,           # 가로 길이
        'Ly': 0.5,           # 세로 높이
        
        # 재질 설정
        'k': 200.0,          # 열전도율 (k) [W/m*K], 예: 알루미늄
        
        # 내부 열원 크기 및 위치 설정 (단위: m)
        'hs_xc': 0.3,        # 열원 중심 X 좌표
        'hs_yc': 0.25,       # 열원 중심 Y 좌표
        'hs_w': 0.15,        # 열원 가로 너비 (W)
        'hs_h': 0.15,        # 열원 세로 높이 (H)
        'hs_Q': 1.5e6,       # 열원 열생성량 (q) [W/m³]
        
        # 경계 온도 설정 (단위: °C)
        'T_left': 100.0,     # 왼쪽 뜨거운 벽면 온도 (Dirichlet)
        'T_right': 20.0,     # 오른쪽 차가운 벽면 온도 (Dirichlet)
        'T_top': 0.0,        # 상단 온도 (fixed인 경우 사용)
        'T_bottom': 0.0,     # 하단 온도 (fixed인 경우 사용)
        
        # 경계 조건 유형 설정 ("adiabatic": 단열, "fixed": 고정온도)
        'top_bc': 'adiabatic',    # 상단 경계조건 (Neumann 단열)
        'bottom_bc': 'adiabatic', # 하단 경계조건 (Neumann 단열)
        
        # 수치해석 제어 매개변수
        'tol': 1e-5,         # 반복 계산 종료 residual 기준 (수렴 오차)
        'max_iter': 15000,   # 최대 반복 계산 횟수 (격자 증가 대응)
        'omega': 1.2,        # SOR 솔버 가속 매개변수 (벡터화 연산 안정성 유지)
        'T_limit': 80.0      # 부품 허용 한계 온도 [°C]
    }
    return inputs


# ==========================================
# 2. 계산 함수 (solve_heat_conduction)
# ==========================================
def solve_heat_conduction(Nx, Ny, params):
    """
    2차원 정상 상태 열전도 편미분 방정식을 FDM(유한차분법)과 Gauss-Seidel 반복법으로 해결하는 함수.
    반복 계산 수렴 오차가 tol 이하가 될 때까지 루프를 수행함.
    
    [FDM 이산화 공식 (dx = dy = h 일 때)]
    T_new[j, i] = 0.25 * (T[j, i+1] + T[j, i-1] + T[j+1, i] + T[j-1, i] + (q[j, i] * h^2 / k))
    """
    Lx = params['Lx']
    Ly = params['Ly']
    k = params['k']
    tol = params['tol']
    max_iter = params['max_iter']
    omega = params['omega']
    
    # 격자 간격 계산
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    beta = dx / dy  # 격자 종횡비
    
    # 좌표 배열 생성
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    
    # 2D 온도장 배열 초기화 (Ny행 x Nx열)
    # 수렴 속도를 높이기 위해 좌우 벽면 사이의 일차원 선형 보간값으로 초기 온도를 설정
    T = np.zeros((Ny, Nx))
    for i in range(Nx):
        t_linear = params['T_left'] + (params['T_right'] - params['T_left']) * (i / (Nx - 1))
        T[:, i] = t_linear
        
    # Dirichlet 경계조건 강제 부여 (좌측 및 우측 벽면 고정)
    T[:, 0] = params['T_left']
    T[:, -1] = params['T_right']
    
    # 내부 사각 열원의 격자 매핑 (체적 발열 밀도 q[j, i] 정의)
    q = np.zeros((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):
            cx = i * dx
            cy = j * dy
            # 해당 격자 노드가 물리적인 사각 열원 범위 안에 들어오는지 판별
            if (params['hs_xc'] - params['hs_w']/2.0 <= cx <= params['hs_xc'] + params['hs_w']/2.0) and \
               (params['hs_yc'] - params['hs_h']/2.0 <= cy <= params['hs_yc'] + params['hs_h']/2.0):
                q[j, i] = params['hs_Q']
                
    # 수치 반복 루프 (Gauss-Seidel 및 SOR 계산 수행)
    beta_sq = beta ** 2
    denom = 2.0 * (1.0 + beta_sq)
    dx_sq = dx ** 2
    
    converged = False
    iteration = 0
    
    for it in range(max_iter):
        T_old = T.copy()
        
        # 1. 하단 경계면 (j = 0) 이산화 식 적용 (Neumann 단열 조건)
        if params['bottom_bc'] == 'adiabatic':
            T_gs = (T[0, 2:] + T[0, :-2] + 2.0 * beta_sq * T[1, 1:-1] + (q[0, 1:-1] * dx_sq / k)) / denom
            T[0, 1:-1] = (1.0 - omega) * T[0, 1:-1] + omega * T_gs
            
        # 2. 내부 노드들 (1 ~ Ny-2) 이산화 식 적용
        for j in range(1, Ny - 1):
            T_gs = (T[j, 2:] + T[j, :-2] + beta_sq * (T[j+1, 1:-1] + T[j-1, 1:-1]) + (q[j, 1:-1] * dx_sq / k)) / denom
            T[j, 1:-1] = (1.0 - omega) * T[j, 1:-1] + omega * T_gs
            
        # 3. 상단 경계면 (j = Ny-1) 이산화 식 적용 (Neumann 단열 조건)
        if params['top_bc'] == 'adiabatic':
            T_gs = (T[-1, 2:] + T[-1, :-2] + 2.0 * beta_sq * T[-2, 1:-1] + (q[-1, 1:-1] * dx_sq / k)) / denom
            T[-1, 1:-1] = (1.0 - omega) * T[-1, 1:-1] + omega * T_gs
            
        # 좌우 경계 온도 항상 Dirichlet 고정
        T[:, 0] = params['T_left']
        T[:, -1] = params['T_right']
        
        # 최대 잔차(residual) 계산하여 수렴 여부 판정
        residual = np.max(np.abs(T - T_old))
        iteration = it + 1
        
        if residual < tol:
            converged = True
            break
            
    return T, x, y, iteration, converged


# ==========================================
# 3. 판정 함수 (perform_judgment)
# ==========================================
def perform_judgment(T, params, iteration, converged):
    """
    계산된 온도장 분포 결과를 공학적으로 판정하고 터미널 콘솔에 분석 요약을 출력하는 함수.
    """
    T_max = np.max(T)
    T_mean = np.mean(T)
    T_limit = params['T_limit']
    
    # 허용 온도를 초과하는 영역의 격자 수 비율 계산
    exceed_count = np.sum(T > T_limit)
    total_nodes = T.size
    exceed_ratio = (exceed_count / total_nodes) * 100.0
    
    print("\n" + "="*50)
    print("           [2D 열전도 공학적 판정 리포트]")
    print("="*50)
    print(f"  - 계산 수렴 여부   : {'성공 (Converged)' if converged else '실패 (Max Iteration 도달)'}")
    print(f"  - 반복 계산 횟수   : {iteration}회 실행")
    print(f"  - 판내 최고 온도   : {T_max:.2f} °C")
    print(f"  - 판내 평균 온도   : {T_mean:.2f} °C")
    print(f"  - 부품 허용 온도   : {T_limit:.1f} °C")
    print(f"  - 허용온도 초과비율: {exceed_ratio:.2f} %")
    print("-"*50)
    
    # 판정 의견 도출
    if T_max > T_limit:
        print("  -> 공학적 판정: [DANGER (경고)]")
        print(f"    열원 근처 최고 온도({T_max:.2f}°C)가 구조물 안전 허용치({T_limit}°C)를 초과합니다.")
        print("    열원 배치를 우측 냉각벽면 근처로 조정하여 방열 설계를 보강하십시오.")
    else:
        print("  -> 공학적 판정: [SAFE (안전)]")
        print(f"    열원 근처 최고 온도({T_max:.2f}°C)가 구조물 안전 허용치({T_limit}°C) 이내로 안전합니다.")
        print("    차가운 벽면으로의 열 전도 전달이 원활하여 안전 설계 요건에 충족합니다.")
    print("="*50 + "\n")
    
    return T_max, T_mean, exceed_ratio


# ==========================================
# 4. 시각화 함수 (plot_results)
# ==========================================
def plot_results(T, x, y, params, title_suffix=""):
    """
    2차원 등온선 Heatmap과 열원 중심을 통과하는 1차원 온도선 그래프를 동시에 생성하는 시각화 함수.
    """
    Lx = params['Lx']
    Ly = params['Ly']
    T_limit = params['T_limit']
    
    # 새로운 그림(figure) 생성 및 다중 플롯 구성 (가로로 2개 배치)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # --- [좌측 플롯] 2D 등온 Heatmap ---
    ax1 = axes[0]
    X, Y = np.meshgrid(x, y)
    
    # Render isothermal Heatmap
    heatmap = ax1.imshow(T, extent=[0, Lx, 0, Ly], origin='lower', cmap='jet', aspect='equal')
    cbar = fig.colorbar(heatmap, ax=ax1)
    cbar.set_label('Temperature (°C)', fontsize=10)
    
    # Isothermal contours overlay
    contours = ax1.contour(X, Y, T, levels=8, colors='black', linewidths=0.5, alpha=0.6)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
    
    # Highlight danger boundary if max temp exceeds limit (T = T_limit)
    if np.max(T) > T_limit:
        try:
            danger_contour = ax1.contour(X, Y, T, levels=[T_limit], colors='red', linewidths=2.0, linestyles='dashed')
            ax1.clabel(danger_contour, inline=True, fontsize=9, fmt='Danger(%.0f°C)', colors='red')
        except Exception:
            pass
            
    # Draw outline of square heat source (dashed rectangle)
    hs_w = params['hs_w']
    hs_h = params['hs_h']
    hs_rect = plt.Rectangle(
        (params['hs_xc'] - hs_w/2.0, params['hs_yc'] - hs_h/2.0),
        hs_w, hs_h,
        fill=False, edgecolor='white', linestyle='--', linewidth=1.5, label='Heat Source'
    )
    ax1.add_patch(hs_rect)
    ax1.legend(loc='upper right')
    
    ax1.set_title(f"2D Temperature Field Heatmap {title_suffix}", fontsize=11, weight='bold')
    ax1.set_xlabel("X-coordinate (m)", fontsize=9)
    ax1.set_ylabel("Y-coordinate (m)", fontsize=9)
    
    # --- [Right Plot] 1D Temperature Profile crossing center line ---
    ax2 = axes[1]
    
    # Find grid index closest to center of the heat source (Y coordinate)
    j_center = np.argmin(np.abs(y - params['hs_yc']))
    y_actual = y[j_center]
    T_profile = T[j_center, :]
    
    ax2.plot(x, T_profile, color='blue', linewidth=2.0, label=f"Temp Profile (Center Y = {y_actual:.3f}m)")
    
    # Limit temperature guideline
    ax2.axhline(y=T_limit, color='red', linestyle='--', linewidth=1.5, label=f"Limit Line ({T_limit}°C)")
    
    # Highlight physical heat source location with soft orange span
    ax2.axvspan(params['hs_xc'] - hs_w/2.0, params['hs_xc'] + hs_w/2.0, color='orange', alpha=0.15, label='Heat Source Area')
    
    ax2.set_title("Temperature Profile across Heat Source Center", fontsize=11, weight='bold')
    ax2.set_xlabel("X Position (m)", fontsize=9)
    ax2.set_ylabel("Temperature (°C)", fontsize=9)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    fig.tight_layout()
    plt.show()


# ==========================================
# 5. 저장 함수 (save_results_to_files)
# ==========================================
def save_results_to_files(T, x, y, params, T_max, T_mean, exceed_ratio, csv_filename="temperature_field.csv", txt_filename="thermal_report.txt"):
    """
    시뮬레이션 해석 결과를 구조화된 CSV 격자 테이블 및 요약 텍스트 보고서 파일로 디렉토리에 저장하는 함수.
    """
    Ny, Nx = T.shape
    
    # 1. 2D 온도 필드 CSV 저장
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 컬럼 헤더 X축 좌표 기입
            writer.writerow(["Y \\ X"] + [f"{x_val:.4f}m" for x_val in x])
            # 행별 데이터 기입 (j 인덱스는 y 좌표)
            for j in range(Ny):
                row_data = [f"{y[j]:.4f}m"] + [f"{T[j, i]:.4f}" for i in range(Nx)]
                writer.writerow(row_data)
        print(f" CSV 격자 데이터 내보내기 성공: {csv_filename}")
    except Exception as e:
        print(f" CSV 저장 실패: {e}")
        
    # 2. 공학 리포트 요약 텍스트 파일 저장
    try:
        with open(txt_filename, mode='w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("      2D 열전도 수치해석 및 격자 민감도 공학 보고서 (주제 B)\n")
            f.write("="*60 + "\n")
            f.write(f"- 계산 격자 수: {Nx} x {Ny} Nodes\n")
            f.write(f"- 판 치수: {params['Lx']}m x {params['Ly']}m\n")
            f.write(f"- 열전도율(k): {params['k']} W/m*K\n")
            f.write(f"- 열원 위치: X_center = {params['hs_xc']}m, Y_center = {params['hs_yc']}m\n")
            f.write(f"- 열원 생성량(Q): {params['hs_Q']:.2e} W/m^3\n")
            f.write("-" * 55 + "\n")
            f.write(f"- 해석 최고 온도(T_max): {T_max:.3f} °C\n")
            f.write(f"- 해석 평균 온도(T_mean): {T_mean:.3f} °C\n")
            f.write(f"- 부품 허용 온도: {params['T_limit']} °C\n")
            f.write(f"- 한계 초과 영역 비율: {exceed_ratio:.2f} %\n")
            f.write("-" * 55 + "\n")
            f.write("- 조별 종합 설계 검토의견:\n")
            if T_max > params['T_limit']:
                f.write(f"  [DANGER - 안전 설계 보완 권고]\n")
                f.write(f"  최고온도가 허용 한계를 {T_max - params['T_limit']:.2f}°C 초과하므로,\n")
                f.write("  발열체의 실장 위치를 차가운 우측 냉각벽면 방향으로 밀착시키거나 k 전도도가 높은 자재 선정을 권고합니다.\n")
            else:
                f.write(f"  [SAFE - 안전 요건 충족]\n")
                f.write("  구조물 온도 제한 요건에 만족하며 열원 배치가 양호하여 작동 안정성을 확보한 설계 상태입니다.\n")
            f.write("="*60 + "\n")
        print(f" [Saved] 텍스트 요약 보고서 내보내기 성공: {txt_filename}")
    except Exception as e:
        print(f" [Error] TXT 저장 실패: {e}")


# ==========================================
# 6. 격자 민감도 해석 루프 함수
# ==========================================
def run_grid_sensitivity_analysis(params):
    """
    Nx = 20, 40, 80 격자 크기를 변경하며 온도 계산을 진행하고,
    격자 크기에 따른 최고 온도 및 평균 온도 변화를 비교 그래프로 시각화하는 자동화 함수.
    """
    print("\n" + "="*50)
    print("       [격자 민감도 자동 분석 루프 구동]")
    print("="*50)
    
    grids = [20, 40, 80]
    t_max_list = []
    t_mean_list = []
    
    for g in grids:
        ny = int(g / 2) # 형상비 보존을 위해 세로는 절반으로 지정
        if ny < 4: ny = 4
        
        print(f"  > Grid {g}x{ny} 계산 중...")
        T, _, _, iters, _ = solve_heat_conduction(g, ny, params)
        
        tm = np.max(T)
        tmean = np.mean(T)
        t_max_list.append(tm)
        t_mean_list.append(tmean)
        print(f"    결과: T_max = {tm:.3f}°C (반복횟수: {iters}회)")
        
    print("="*50)
    
    # Grid independence analysis graph
    plt.figure(figsize=(7, 4.5))
    plt.plot(grids, t_max_list, marker='o', color='red', linewidth=2, label='Max Temperature (T_max)')
    plt.plot(grids, t_mean_list, marker='s', color='blue', linewidth=1.5, linestyle='--', label='Mean Temperature (T_mean)')
    
    # Annotate values above points
    for x, y in zip(grids, t_max_list):
        plt.annotate(f"{y:.2f}°C", (x, y), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9, weight='bold', color='red')
        
    plt.title("Grid Sensitivity & Temperature Convergence", fontsize=11, weight='bold')
    plt.xlabel("Number of Grid Nodes (Nx)", fontsize=9)
    plt.ylabel("Calculated Temperature (°C)", fontsize=9)
    plt.legend(loc='center right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. 도전과제: 열원 위치 스윕 분석 함수
# ==========================================
def run_heat_source_sweep_analysis(params):
    """
    도전 과제 해결: 내부 사각 열원의 수평 좌표(Xc)를 좌측(0.15m)에서 우측(0.85m)으로 순차적으로
    스윕하며 해석을 수행하고, Xc 위치 대비 최고 온도의 하강 효과를 분석하여 선도로 나타냅니다.
    """
    print("\n" + "="*50)
    print("    [도전 과제: 열원 실장 위치에 따른 감쇠 스윕]")
    print("="*50)
    
    # 0.15m에서 0.85m까지 6단계로 수평 좌표 스윕 범위 설정
    x_positions = np.linspace(0.15, 0.85, 6)
    t_max_sweep = []
    
    # 격자 크기는 균일한 40x20 격자 채택
    Nx, Ny = 40, 20
    
    # 기존 원본 파라미터 보존용 임시 딕셔너리
    temp_params = params.copy()
    
    for idx, x_pos in enumerate(x_positions):
        temp_params['hs_xc'] = x_pos
        print(f"  > [{idx+1}/6단계] 열원 위치 Xc = {x_pos:.3f}m 계산 중...")
        T, _, _, _, _ = solve_heat_conduction(Nx, Ny, temp_params)
        tm = np.max(T)
        t_max_sweep.append(tm)
        print(f"    최고 온도 T_max = {tm:.2f}°C")
        
    print("="*50)
    
    # Heat source placement sensitivity graph
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_positions, t_max_sweep, marker='D', color='orange', linewidth=2, label='Max Temp vs. Heat Source Position')
    plt.axhline(y=params['T_limit'], color='red', linestyle='--', label=f"Structural Safety Limit ({params['T_limit']}°C)")
    
    # Add arrow annotation for trend
    plt.annotate("Strong conduction heat dissipation\nnear cold wall (T=20°C)", 
                 xy=(x_positions[-1], t_maxs := t_max_sweep[-1]), 
                 xytext=(x_positions[-3], t_maxs + 15),
                 arrowprops=dict(facecolor='blue', shrink=0.1, width=1.5, headwidth=6, edgecolor='none'),
                 fontsize=9, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='orange'))
                 
    plt.title("Max Temperature vs. Heat Source Center X-coordinate (Xc)", fontsize=11, weight='bold')
    plt.xlabel("Heat Source Center X Position (m)", fontsize=9)
    plt.ylabel("Domain Max Temperature T_max (°C)", fontsize=9)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==========================================
# 8. 통합 실행 메인 함수
# ==========================================
def main():
    print("*"*60)
    print("   2D HEAT SOLVER & GRID SENSITIVITY SYSTEM ")
    print("*"*60)
    
    # 1단계: 해석 매개변수 및 설계 조건 가져오기
    params = get_inputs()
    
    # 2단계: 기준 조건 해석 실행 (기본 40x20 격자 설정)
    Nx_base = 80
    Ny_base = 40
    print(f"\n[Baseline Run] 표준설계 해석 실행 중... (격자: {Nx_base}x{Ny_base})")
    
    T, x, y, iteration, converged = solve_heat_conduction(Nx_base, Ny_base, params)
    
    # 3단계: 해석 데이터에 기반한 공학적 판정 리포팅 출력
    T_max, T_mean, exceed_ratio = perform_judgment(T, params, iteration, converged)
    
    # 4단계: 기준 해석 결과 2D Heatmap & 1D Profile 도식 시각화
    print("[Plotting] 수치해석 그래픽 가시화 차트를 그리는 중...")
    plot_results(T, x, y, params, title_suffix="(Base 40x20 Grid)")
    
    # 5단계: 격자 데이터 CSV 및 보고서 요약 TXT 저장
    print("[File Saving] 로컬 출력 파일 생성 중...")
    save_results_to_files(T, x, y, params, T_max, T_mean, exceed_ratio)
    
    # 6단계: 필수 과제 요건 - 격자 독립성 민감도 자동 루프 분석 구동
    run_grid_sensitivity_analysis(params)
    
    # 7단계: 도전 과제 요건 - 열원 위치 스윕을 통한 최고 온도 억제 분석 구동
    run_heat_source_sweep_analysis(params)
    
    print("\n" + "*"*60)
    print("   모든 Term Project 시뮬레이션 및 데이터 파일 저장이 종료되었습니다!")
    print("   작업 폴더의 CSV 데이터와 TXT 요약본을 보고서에 첨부하십시오.")
    print("*"*60)


if __name__ == "__main__":
    main()
