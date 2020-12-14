/**
 * @Author: Dragan Rangelov <uqdrange>
 * @Date:   17-7-2019
 * @Email:  d.rangelov@uq.edu.au
 * @Last modified by:   uqdrange
 * @Last modified time: 06-8-2019
 * @License: CC-BY-4.0
 */



% adding circ stats stuff and dividing stuff out by cue as well

clear all;
addpath('/Users/uqdlevyb/Desktop/02_Data/02_Data_COSMOS/Behav_ALL/CircStat');

%% READING IN THE DATA SPREADSHEET
% fname = [cd filesep 'BHV_V2.xlsx'];
fname = '/Users/uqdlevyb/Desktop/02_Data/02_Data_COSMOS/Behav_ALL/BHV_V2.xlsx';

sheet_name = 'ALL';
BHV_V2 = xls2struct(fname);
fields = fieldnames(BHV_V2);
for field = 1:length(fields)
    BHV_V2.(fields{field}) = cell2mat(BHV_V2.(fields{field}));
end;
%% PLAYING WITH THE DATA
figure
id_sbj = unique(BHV_V2.subjectID);
n_sbj = length(id_sbj);
n_tasks = length(unique(BHV_V2.task));
n_cue = length(unique(BHV_V2.cue));
K = nan(n_sbj,n_tasks); % gaussian (von mises) distribution capturing the variability of the responses
pT = nan(n_sbj,n_tasks); % precision target parameter (defined as pT is the probability of responding with the target value)
pN = nan(n_sbj,n_tasks); % precision non target parameter (swap errors?) (defined as the probability of responding with a non-target value)
pU = nan(n_sbj,n_tasks); % random (guesses) (defined as the probability of responding "randomly")
idx = 0;
for task = 1:n_tasks
     for cno = 1:n_cue
        for sno = 1:n_sbj
            idx_sbj = BHV_V2.subjectID == id_sbj(sno); % selecting the participant
            idx_task = BHV_V2.task == task; % selecting the task we want to fit
            idx_good_trl = BHV_V2.fixated == 1; % selecting only good trials
            idx_cue = BHV_V2.cue == cno; % selecting the correct cue (1/2)
            idx_tot = idx_good_trl & idx_task & idx_sbj & idx_cue;
            T = BHV_V2.target(idx_tot);
            T_rad = wrapToPi(T);
            X = BHV_V2.response(idx_tot);
            X_rad = wrapToPi(X);
            [B, LL] = CO16_fit(X_rad,T_rad);
            K(sno,task) = B(1);
            pT(sno,task) = B(2);
            pN(sno,task) = B(3);
            pU(sno,task) = B(4);
            errMagn = angle(exp(1i*(X_rad - T_rad)));
            idx = idx + 1;
            subplot(6,n_sbj,idx)
            tmp = histogram(errMagn, 'BinEdges', linspace(-pi, pi, 12), 'Normalization', 'probability');

            % this was from the non-cue stuff
            % allProportions(sno,task,:) = tmp.Values;
            % now puting cue info into the same array, but ordered T1C1,
            % T1C2, T2C1, T2C2 .... instead of just T1 T2 T3
            allProportions(sno,(task-1)*2+cno,:) = tmp.Values;


            % in here we need to actually store all the error magnitude data into a variable, BHV_V2.errMagn
            % idx tot is a bunch of 0's and 1's that says which trials are good
            BHV_V2.errMagn(idx_tot) = errMagn;
            BHV_V2.errMagn = BHV_V2.errMagn'; % Here, the values are 1x22448 and we want one column with several raw (22448 x 1) this line transposes them



        end
    end
end

    % figure behavioural
    figure
    gavHist = squeeze(mean(allProportions, 1));
    for task = 1:n_tasks*n_cue

        subplot(1,6,task); % the 3 turned to a 6 as now have task x cue
        bar(gavHist(task,:))
    end


%% this part writes trial back to the spreadsheet for the error magnitude (should be one value per row)
%
%             %  read data and only use the RAW
%             [NUM,TXT,RAW] = xlsread([fname],1);
%             RAW_new = RAW;
%
%             % write data
%             iX = 22; % column 22 in the spreadsheet is where the data goes
%             header_rows = 1; % maybe there's 1... you will need to check
%             for iY = 1:n_trials     % trials
%                 RAW_new(iY+header_rows, iX) = {BHV_V2.errMagn(iY)};
%             end
%             BHV_V2.errMagn = BHV_V2.errMagn';
%             xlswrite('BHV_V2.xlsx' ,RAW_new)
%
%% circular statistics on ErrMagn (mean and SD) via circ_stats function
%         define path for function
rowno = 0;
for task = 1:n_tasks
    for cno = 1:n_cue
        for sno = 1:n_sbj
            idx_sbj = BHV_V2.subjectID == id_sbj(sno); % selecting the participant
            idx_task = BHV_V2.task == task; % selecting the task we want to fit
            idx_good_trl = BHV_V2.fixated == 1; % selecting only good trials
            idx_cue = BHV_V2.cue == cno; % selecting the correct cue (1/2)
            idx_tot = idx_good_trl & idx_task & idx_sbj & idx_cue;

            errMagn_sub_cond_good = BHV_V2.errMagn(idx_tot); % contains the data for the correct subj no, the correct cond, the good trials
            stats = circ_stats(errMagn_sub_cond_good);

            % sno task cue   mean median var std std0 skewness skewness0 kurtosis kurtosi0
            rowno = rowno+1;
            OUTPUT.stats(rowno,:) = [id_sbj(sno) task cno stats.mean stats.median stats.var stats.std stats.std0 stats.skewness stats.skewness0 stats.kurtosis stats.kurtosis0];

        end
    end
end
OUTPUT.colnames = { 'sno' 'task' 'cue' 'mean' 'median' 'var' 'std' 'std0' 'skewness' 'skewness0' 'kurtosis' 'kurtosis0'};

%% function for complex number (decision-weight analysis)
rowno = 0;
for task = 1:n_tasks
    for cno = 1:n_cue
        for sno = 1:n_sbj
            idx_sbj = BHV_V2.subjectID == id_sbj(sno); % selecting the participant
            idx_task = BHV_V2.task == task; % selecting the task we want to fit
            idx_good_trl = BHV_V2.fixated == 1; % selecting only good trials
            idx_cue = BHV_V2.cue == cno; % selecting the correct cue (1/2)
            idx_tot = idx_good_trl & idx_task & idx_sbj & idx_cue; % trials for this process loop

            % first thing we have to do is get the Predictor data  - which
            % parts of the BHV structrure we need will be based on the task
            % type
            % only do for sensible things.
            P = zeros(sum(idx_tot),6);
            switch task
                case {1 2}
                    P(:,1)=BHV_V2.ori_c_A(idx_tot);
                    P(:,2)=BHV_V2.ori_c_B(idx_tot);
                    P(:,3)=BHV_V2.ori_c_C(idx_tot);
                    P(:,4)=BHV_V2.ori_u_A(idx_tot);
                    P(:,5)=BHV_V2.ori_u_B(idx_tot);
                    P(:,6)=BHV_V2.ori_u_C(idx_tot);
                case 3
                    P(:,1)=BHV_V2.loc_c_A(idx_tot);
                    P(:,2)=BHV_V2.loc_c_B(idx_tot);
                    P(:,3)=BHV_V2.loc_c_C(idx_tot);
                    P(:,4)=BHV_V2.loc_u_A(idx_tot);
                    P(:,5)=BHV_V2.loc_u_B(idx_tot);
                    P(:,6)=BHV_V2.loc_u_C(idx_tot);
            end
            C = zeros(sum(idx_tot),1);
            C = BHV_V2.response(idx_tot);

            % make data complex
            Pcomplex = 1*cos(P) + 1i*1*sin(P);
            Ccomplex = 1*cos(C) + 1i*1*sin(C);

            % make complex conjugate
            % Pcomplexconj = conj(Pcomplex)'; % conj and transpose
            % DRAGAN changed this
            % according to https://www.mathworks.com/help/matlab/ref/ctranspose.html
            % using transpose operator "''" over a matrix that contains complex numbers
            % automatically computes conjugate transpose
            % so, the original formula has performed conjugate twice
            Pcomplexconj = Pcomplex';

            % do the complex regression
            Answer = inv(Pcomplexconj*Pcomplex) * (Pcomplexconj * Ccomplex);

            % store in output array
            % sno task cue  Answer
            rowno = rowno+1;
            OUTPUT.DecWeight(rowno,:) = [id_sbj(sno) task cno Answer'];


                                        % ################################ NOTES ##################################
                                        % ################################ NOTES ##################################
                                        % ################################ NOTES ##################################
                                        % %
                                        % % def cmplxRegression(crit, pred):
                                        % %     coefs = (np.asmatrix(np.asmatrix(pred).H)
                                        % %                         * np.asmatrix(pred)).I * (np.asmatrix(pred).H
                                        % %                                                   * np.asmatrix(crit))
                                        % %    return np.array(coefs)
                                        % %
                                        % %
                                        % %    np.asmatrix(pred) = X
                                        % %    np.asmatrix(crit) = Y
                                        % % np.asmatrix(np.asmatrix(pred).H) = Xcc
                                        % %
                                        % %   inv(Xcc X ) (Xcc Y)
                                        %
                                        %
                                        %
                                        % % P starting predictor data
                                        % % C criteria data
                                        %
                                        % % to make real angle data into complex numbers
                                        % % P real data in radians
                                        % % C criteria data in radians
                                        % % A  ampltiude = 1
                                        % % Pcomplex complex data
                                        %
                                        % Pcomplex = 1*cos(P) + 1i*1*sin(P);
                                        % Ccomplex = 1*cos(C) + 1i*1*sin(C);
                                        %
                                        % % Complex conjugate
                                        % % CC = complex conjugate
                                        %
                                        % Pcomplexconj = conj(Pcomplex)'; % conj and transpose
                                        %
                                        % % so now, the answer......? maybe is
                                        % % Answer = inv(Xcc X ) (Xcc Y)
                                        %
                                        % Answer = inv(Pcomplexconj*Pcomplex) * (Pcomplexconj * Ccomplex)
                                        %


        end
    end
end

%% Process the decision weight info to get out theta/angle and amplitude/length

%OUTPUT.theta1 = atan(imag(OUTPUT.DecWeight)./real(OUTPUT.DecWeight)); % this matches the notes you have
OUTPUT.theta = angle(OUTPUT.DecWeight);
OUTPUT.theta_cos = cos(OUTPUT.theta);
OUTPUT.theta(:,1:3) = OUTPUT.DecWeight(:,1:3); % copy first three columns (subj,task,cue) data and put into the angle array
OUTPUT.theta_cos(:,1:3) = OUTPUT.DecWeight(:,1:3);

%OUTPUT.length1 = sqrt(real(OUTPUT.DecWeight).^2 + imag(OUTPUT.DecWeight).^2);  % this matches the notes you have
OUTPUT.length = abs(OUTPUT.DecWeight); % should already have subj.task,cue left alone here (cos its a length calc not an angle calc)

%% figure length
rownum = 0
for task = 1:n_tasks
    for sno = 1:n_sbj
        idx_sbj_2 = OUTPUT.length == id_sbj(sno); % selecting the participant
            idx_task = BHV_V2.task == task; % selecting the task we want to fit
            idx_good_trl = BHV_V2.fixated == 1; % selecting only good trials
            idx_cue = BHV_V2.cue == cno; % selecting the correct cue (1/2)
            idx_tot = idx_good_trl & idx_task & idx_sbj & idx_cue; % trials for this process loop

            mean_length= OUTPUT.length(:,4:9)

         %store in output array
            % sno task cue  length
            rownum = rownum+1;
            OUTPUT.length(rownum,:) = [id_sbj(sno) task cno mean_length];
    end
end

histogram(OUTPUT.length)

figure

    gavHist2 = squeeze(mean(OUTPUT.length, 1));
    for task = 1:n_tasks

        subplot(3,1,task); % the 3 turned to a 6 as now have task x cue
        bar(gavHist2(task,:))
    end
