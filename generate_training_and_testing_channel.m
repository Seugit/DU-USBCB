close all
clear all
warning('off','all');

M=4;  %天线面板水平单元数
N=1;  %天线面板垂直单元数
MN = M*N; %天线数
R=1;  %用户天线数
BW=20*1e6; %带宽（用于时域信道转频域）
no_ca=1;   %载波数（用于时域信道转频域）


% 基线天线参数设置
a2 = qd_arrayant( '3gpp-3d', M, N, [], 1, [], 0.5 );
a2.combine_pattern;                         % Calculate array response
a2.element_position(1,:) = 0.5;             % Distance from pole in [m]
%用户天线参数设置
aMT = qd_arrayant('omni');                  % MT antenna configuration
%aMT.copy_element(1,R);
%aMT.Fa(:,:,2) = 0;
%aMT.Fb(:,:,2) = 1;

%%%%%%%
%%%%%%%
type1 = 'training';
%type1 = 'test';
if strcmp(type1, 'training')
    nr_realizations = 24*500;
    no_rx = 12;                     % Number of MTs (directly scales the simulation time)
    no_rx1 = 6;                     % 选择出来的用户数
    no_BS = 2;                     % 基站数，在基站模式为“regular”时，默认三个扇区，扇区数为3*no_BS
    dis_BS = 50;
    dis_user_max = 80;
    dis_user_min = 20;
    tx_position_x = [-50,50];
    tx_position_y = [0,0];
    indoor = 0.0;
    track_length = 0;
    
elseif strcmp(type1, 'test')
    nr_realizations = 500;
    no_rx = 10;                     % Number of MTs (directly scales the simulation time)
    no_rx1 = no_rx;                     % 选择出来的用户数
    no_BS = 7;                     % 基站数，在基站模式为“regular”时，默认三个扇区，扇区数为3*no_BS
    dis_BS = 50;                   % 基站间距
    dis_user_max = 100;
    dis_user_min = 0.1;
    indoor = 0.0;
    track_length = 0;
end


select_scenario = 1;                      
select_fequency = 1;                     

s = qd_simulation_parameters;               % Set general simulation parameters
s.center_frequency = [2.1e9];   % Set center frequencies for the simulations
s.center_frequency = s.center_frequency( select_fequency );
no_freq = numel( s.center_frequency );

s.use_3GPP_baseline = 1;                    % Disable spherical waves
s.show_progress_bars = 0;                   % Disable progress bar

isd = [ 100, 150, 200 ];                    % 站间距
no_go_dist_min = [ 10, 35, 5 ];                 % 最小离站距离
no_go_dist_max = [50, 75,100, 200];
%配置基站信息
%dis_BS = 50; 
l(1,1) = qd_layout.generate( 'regular', no_BS, dis_BS, a2);     %'regular':每个基站有3个扇区，‘hexagonal’一个小区一个扇    
l(1,1).simpar = s;                                              % Set simulation parameters
if strcmp(type1, 'training')
    l(1,1).tx_position(1,:) = tx_position_x;
    l(1,1).tx_position(2,:) = tx_position_y;
end
l(1,1).tx_position(3,:) = 25;                                   % 25 m BS height
l(1,1).name = 'UMa';

tic
clear c



H_realizations = zeros(nr_realizations,no_rx1,M*N*no_BS*3);
i_rea = 1;
while i_rea <= nr_realizations
% 配置用户位置信息
    i_rea
    for il = select_scenario                                        % Drop users in each layout
        l(1,il).no_rx = no_rx;                                      % Number of users

        ind = true( 1,no_rx );                                  % UMa / UMi placement
        ind2 = true;
        %dis_user = isd(1); % 用户最远距离
        l(1,il).randomize_rx_positions(dis_user_max, 1.5, 1.5, track_length, [], dis_user_min, [] );
        floor = randi(5,1,l(1,il).no_rx) + 3;                   % Number of floors in the building
        for n = 1 : l(1,il).no_rx
            floor( n ) =  randi(  floor( n ) );                 % Floor level of the UE
        end
        l(1,il).rx_position(3,:) = 3*(floor-1) + 1.5;           % Height in meters

        % Set the scenario and assign LOS probabilities (80% of the users are indoor)

        indoor_rx = l(1,il).set_scenario('3GPP_38.901_UMa',[],[],indoor); %室内用户占比
        l(1,il).rx_position(3,~indoor_rx) = 1.5;            % Set outdoor-users to 1.5 m height 室外用户高度为1.5米
        l(1,il).rx_array = aMT;                                     % MT antenna setting
    end
    
    if i_rea < 2
        figure()
        scatter(l(1,il).rx_position(1,:),l(1,il).rx_position(2,:))
        hold on
        scatter(l(1,il).tx_position(1,:),l(1,il).tx_position(2,:),"^")
        xlabel('x/m')
        ylabel('y/m')
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%生成信道

    for il = select_scenario
        b = l(1,il).init_builder;                       % Generate builders

        sic = size( b );
        for ib = 1 : numel(b)
            [ i1,i2 ] = qf.qind2sub( sic, ib );
            scenpar = b(i1,i2).scenpar;                 % Read scenario parameters
            scenpar.SC_lambda = 0;                      % Disable spatial consistency of SSF
            b(i1,i2).scenpar_nocheck = scenpar;         % Save parameters without check (faster)
        end

        b = split_multi_freq( b );                      % Split the builders for multiple frequencies
        gen_parameters( b );                            % Generate LSF and SSF parameters (uncorrelated)
        cm = get_channels( b );                         % Generate channels

        cs = split_tx( cm, {1:M*N,M*N+1:2*M*N,2*M*N+1:3*M*N} );         % Split sectors for Antenna configuration 2
        c = qf.reshapeo( cs, [ no_rx, l(1,il).no_tx*3, no_freq ] );
    end
    %生成频域信道
    H=zeros(no_rx,M*no_BS*3);
    for i=1:no_rx
        for q=1:no_BS*3
            H(i,(q-1)*M*N+1:q*M*N)=c(i,q).fr(BW,no_ca);       %频域信号
        end
    end
%     HH = reshape(H,no_rx,no_BS*3,M); 
%     HH_norm = zeros(no_rx,no_BS*3);
%     for rx=1:no_rx
%         for bs = 1:no_BS*3
%             tmp = reshape(HH(rx,bs,:),1,M);
%             HH_norm(rx,bs) = norm(tmp);
%         end
%     end            
    tmp = diag(H*H');
    [a, idx] = sort(tmp);
    if strcmp(type1, 'training')
    	if max(a(no_rx-no_rx1+1:no_rx)) / min(a(no_rx-no_rx1+1:no_rx)) < 10 
            H_realizations(i_rea,:,:) = H(idx(no_rx-no_rx1+1:no_rx),:,:);
            i_rea = i_rea +1;
        end
    elseif strcmp(type1, 'test')
        H_realizations(i_rea,:,:) = H(idx(no_rx-no_rx1+1:no_rx),:,:);
        i_rea = i_rea +1;
    end
end
no_tx = no_BS * 3; 
no_rx = no_rx1;
if strcmp(type1, 'training')
    filename = ['C:\Users\10189\Desktop\',num2str(no_tx),'tx',num2str(no_rx),'rx',num2str(M),'ants',num2str(dis_BS),'disBS',num2str(dis_user_min),'-', num2str(dis_user_max), 'dis_user',num2str(indoor),'indoor_H_realizations_training.mat']
    save(filename,'H_realizations', 'no_rx','no_tx','M','dis_BS','dis_user_min','dis_user_max','indoor')
elseif strcmp(type1, 'test')
     filename = ['C:\Users\10189\Desktop\',num2str(no_tx),'tx',num2str(no_rx),'rx',num2str(M),'ants',num2str(dis_BS),'disBS',num2str(dis_user_min),'-', num2str(dis_user_max), 'dis_user',num2str(indoor),'indoor__H_realizations_test.mat']
    save(filename,'H_realizations', 'no_rx','no_tx','M','dis_BS','dis_user_min','dis_user_max','indoor')
end
