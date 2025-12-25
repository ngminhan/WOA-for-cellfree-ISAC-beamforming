function [F_star, feasible, SSNR_opt] = opt_jsc_WOA(H_comm, sigmasq_comm, gamma_req, sensing_beamsteering, sens_streams, sigmasq_sens, P_all, max_iter, search_agents, F_init)
    % opt_jsc_WOA: Whale Optimization Algorithm - Advanced Version
    % Sử dụng cơ chế Repair và Deb's Rules để xử lý ràng buộc.

    if nargin < 8, max_iter = 100; end
    if nargin < 9, search_agents = 30; end

    [U, M, N] = size(H_comm);
    H_st = reshape(H_comm, U, []); 
    num_antennas = M * N;
    num_streams = U + sens_streams;
    dim = num_antennas * num_streams;

    % Chuẩn bị ma trận Sensing
    a = reshape(sensing_beamsteering, 1, []);
    A_sens = a' * a;
    
    D_matrices = cell(M, 1);
    for m = 1:M
        diag_idx = (m-1)*N + (1:N);
        D_tmp = zeros(num_antennas, num_antennas);
        D_tmp(diag_idx, diag_idx) = eye(N);
        D_matrices{m} = D_tmp;
    end

    % --- 1. Initialization (Khởi tạo tập trung) ---
    X = zeros(search_agents, dim);
    
    % Warm Start: Sử dụng kết quả tốt từ RZF/SDP
    if nargin >= 10 && ~isempty(F_init)
        X(1, :) = reshape(F_init, 1, []);
        % Tạo các cá thể xung quanh Warm Start với bán kính nhỏ (Local Search)
        for i = 2:search_agents
            perturbation = (randn(1, dim) + 1i*randn(1, dim));
            % Bán kính tìm kiếm giảm dần theo i để đa dạng hóa
            scale = 0.1 * (i / search_agents); 
            X(i, :) = X(1, :) + perturbation * scale * sqrt(P_all/num_streams);
        end
    else
        % Random nếu không có warm start
        X = (randn(search_agents, dim) + 1i * randn(search_agents, dim)) * sqrt(P_all / num_streams);
    end

    % Đánh giá quần thể ban đầu
    Leader_pos = X(1, :);
    [Leader_obj, Leader_vio] = evaluate_solution(Leader_pos, H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
    
    for i = 2:search_agents
        [obj, vio] = evaluate_solution(X(i,:), H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
        % So sánh với Leader hiện tại bằng Deb's Rules
        if is_better(obj, vio, Leader_obj, Leader_vio)
            Leader_pos = X(i, :);
            Leader_obj = obj;
            Leader_vio = vio;
        end
    end

    % --- 2. Main Loop ---
    for t = 1:max_iter
        a_woa = 2 - t * (2 / max_iter); % Giảm tuyến tính
        a2 = -1 + t * ((-1)/max_iter);
        
        for i = 1:search_agents
            r1 = rand(); r2 = rand();
            A_vec = 2 * a_woa * r1 - a_woa;
            C_vec = 2 * r2;
            b = 1; l = (a2-1)*rand()+1; 
            p = rand();
            
            X_new = X(i, :);
            
            if p < 0.5
                if abs(A_vec) < 1
                    % Encircling Prey (Tập trung khai thác quanh Leader)
                    D_x = abs(C_vec * Leader_pos - X(i, :)); 
                    X_new = Leader_pos - A_vec * D_x;
                else
                    % Search for Prey (Thám hiểm toàn cục - Cẩn thận phá vỡ cấu trúc tốt)
                    % Thay vì chọn ngẫu nhiên hoàn toàn, ta chọn ngẫu nhiên trong top 50%
                    rand_idx = floor((search_agents/2) * rand() + 1);
                    X_rand = X(rand_idx, :);
                    D_x = abs(C_vec * X_rand - X(i, :));
                    X_new = X_rand - A_vec * D_x;
                end
            else
                % Spiral Update (Di chuyển xoắn ốc)
                dist_to_leader = abs(Leader_pos - X(i, :));
                X_new = dist_to_leader * exp(b .* l) .* cos(2 * pi * l) + Leader_pos;
            end
            
            % --- CRITICAL: Power Repair ---
            % Ngay lập tức sửa lỗi công suất sau khi di chuyển
            X_new = repair_power_constraint(X_new, P_all, D_matrices, num_antennas, num_streams, M);
            X(i, :) = X_new;
        end
        
        % Đánh giá lại toàn bộ quần thể
        for i = 1:search_agents
            [obj, vio] = evaluate_solution(X(i,:), H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
            
            % Cập nhật Leader
            if is_better(obj, vio, Leader_obj, Leader_vio)
                Leader_pos = X(i, :);
                Leader_obj = obj;
                Leader_vio = vio;
            end
        end
        
        % (Optional) Hiển thị tiến trình
        % fprintf('Iter %d: Obj=%.2f, Vio=%.4f\n', t, Leader_obj, Leader_vio);
    end

    % --- 3. Final Result ---
    F_star = reshape(Leader_pos, num_antennas, num_streams);
    
    % Kiểm tra lần cuối
    [SSNR_opt, violation] = evaluate_solution(Leader_pos, H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
    
    % Chấp nhận sai số nhỏ (1e-4) cho khả thi
    feasible = (violation < 1e-4); 
end

% --- Helper Functions ---

function X_repaired = repair_power_constraint(X_vec, P_max, D_mats, num_ant, num_streams, M)
    % Hàm này ép buộc vector X thỏa mãn ràng buộc tổng công suất per-AP
    % Nếu AP nào vượt quá công suất, scale down toàn bộ beam của AP đó.
    
    F = reshape(X_vec, num_ant, num_streams);
    F_cov = F * F'; % Covariance matrix
    
    scaling_factors = ones(M, 1);
    needs_repair = false;
    
    for m = 1:M
        % Tính công suất hiện tại của AP m
        p_current = real(trace(D_mats{m} * F_cov));
        if p_current > P_max
            scaling_factors(m) = sqrt(P_max / p_current);
            needs_repair = true;
        end
    end
    
    if needs_repair
        % Áp dụng scaling cho các dòng tương ứng với anten của AP m
        % F là (M*N) x Streams
        N = num_ant / M;
        for m = 1:M
            if scaling_factors(m) < 1
                row_start = (m-1)*N + 1;
                row_end = m*N;
                F(row_start:row_end, :) = F(row_start:row_end, :) * scaling_factors(m);
            end
        end
        X_repaired = reshape(F, 1, []);
    else
        X_repaired = X_vec;
    end
end

function [obj_val, violation_val] = evaluate_solution(X_vec, H, sigma_sq, gamma, A, sigma_sens, ~, D_mats, U, num_streams)
    % 1. Tính Objective (Sensing SNR)
    F = reshape(X_vec, [], num_streams);
    F_sum = F * F';
    
    obj_sensing = 0;
    for m = 1:length(D_mats)
        obj_sensing = obj_sensing + trace(D_mats{m} * A * D_mats{m} * F_sum);
    end
    obj_val = real(obj_sensing * sigma_sens);

    % 2. Tính Violation (Chỉ tính SINR, vì Power đã được Repair)
    violation_val = 0;
    for u = 1:U
        h_u = H(u, :);
        f_u = F(:, u);
        signal = abs(h_u * f_u)^2;
        inter = 0;
        for k = 1:num_streams
            if k ~= u
                inter = inter + abs(h_u * F(:, k))^2;
            end
        end
        
        sinr_actual = signal / (inter + sigma_sq);
        
        % Nếu SINR < Gamma -> Cộng dồn lỗi
        if sinr_actual < gamma
            % Dùng relative violation để chuẩn hóa
            violation_val = violation_val + (gamma - sinr_actual)/gamma;
        end
    end
end

function result = is_better(obj_new, vio_new, obj_old, vio_old)
    % Deb's Rules for Constraint Handling
    
    epsilon = 1e-4; % Ngưỡng sai số chấp nhận được
    
    is_feas_new = (vio_new <= epsilon);
    is_feas_old = (vio_old <= epsilon);
    
    if is_feas_new && is_feas_old
        % Cả 2 đều khả thi -> Chọn cái Objective cao hơn
        result = (obj_new > obj_old);
    elseif is_feas_new && ~is_feas_old
        % Mới khả thi, cũ không -> Mới tốt hơn
        result = true;
    elseif ~is_feas_new && is_feas_old
        % Mới không khả thi, cũ khả thi -> Mới tệ hơn
        result = false;
    else
        % Cả 2 đều không khả thi -> Chọn cái vi phạm ít hơn
        result = (vio_new < vio_old);
    end
end