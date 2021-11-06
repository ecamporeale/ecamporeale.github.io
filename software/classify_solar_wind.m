% Copyright 2017 Enrico Camporeale
% Licensed under a FreeBSD License (see below)


% Example of how to use the Gaussian Process model described in
% Camporeale et al. (2017) "Solar Wind Classification with Machine Learning" J. Geophys. Res. 
% to classify a whole year from the OMNI2 database


clear all

load parameters_classification.mat ; % load hypers and training_set

run ~/gpml/startup ; % load your local gpml library. See http://www.gaussianprocess.org/gpml/code

urlwrite('ftp://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_2017.dat','omni2_2017.dat'); % Download the OMNI2_2017 file 
omni=load('omni2_2017.dat'); % load the whole year (until available).

meanfunc = {@meanZero};           % empty: don't use a mean function
covfunc = {'covSum', {'covSEiso',{@covPPard, 1}}};  
likfunc = @likLogistic;  

% purge the gaps in omni database
f=find(omni(:,25)==9999);omni(f,:)=[];
f=find(omni(:,23)==9999999);omni(f,:)=[];
f=find(omni(:,24)==999.9);omni(f,:)=[];
f=find(omni(:,30)==9999999);omni(f,:)=[];
f=find(omni(:,9)==999.9);omni(f,:)=[];

disp(['Classifying ' num2str(size(omni,1)) ' events from OMNI']);

Vsw = omni(:,25);
B = omni(:,9);
Tp = omni(:,23)/1.1604e4; % temperature in eV
np = omni(:,24);
sigma_Tp=omni(:,30);
f107 = omni(:,51);
R=omni(:,40);
Texp = (Vsw/258).^3.113;
Va = 21.8*B./sqrt(np);
Sp = Tp./(np).^(2/3);
Tratio = Texp./Tp;

test_set = [Vsw Va Sp Tratio sigma_Tp f107 R];
omni_date = omni(:,1:3);

% normalize test_set
m_te = repmat(m_xtr,size(test_set,1),1); 
s_te = repmat(s_xtr,size(test_set,1),1);
test_set = (test_set-m_te)./s_te;

% classification is performed as '1 vs all'
% warning: the training set xtr is rather large (about 9000 events x 7 attributes)
% the following computations can take some time

disp('Ejecta...')
[ymu, ys2] = gp(hyp_ejecta, @infLaplace, meanfunc, covfunc, likfunc, xtr, ytr_ejecta, test_set);  % predict
prob_ej=.5*(1+ymu);

disp('Coronal holes origin...')
[ymu, ys2] = gp(hyp_coronal_holes, @infLaplace, meanfunc, covfunc, likfunc, xtr, ytr_coronal_holes, test_set);  % predict
prob_ch=.5*(1+ymu);

disp('Sector reversal origin...')
[ymu, ys2] = gp(hyp_sector_reversal, @infLaplace, meanfunc, covfunc, likfunc, xtr, ytr_sector_reversal, test_set);  % predict
prob_sr=.5*(1+ymu);

disp('Streamer belts origin...')
[ymu, ys2] = gp(hyp_streamer_belts, @infLaplace, meanfunc, covfunc, likfunc, xtr, ytr_streamer_belts, test_set);  % predict
prob_sb=.5*(1+ymu);

% renormalize probabilities
sum_probs = prob_ej + prob_ch + prob_sr + prob_sb; 

final_prob = [prob_ej prob_ch prob_sr prob_sb]./sum_probs;

classification_result=[omni_date final_prob]; % this vector contains the time (year-doy-hour) 
                                              % and the probabilities to belong to each category:
                                              %  ejecta - coronal holes origin - sector reversal origin - streamer belt origin
                                              

                                              
                                              
                                              
                                            
                                            
%%%%%%%%%%% COPYRIGHT NOTE %%%%%%%%%%%%%%%%%%

% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

% THIS SOFTWARE IS PROVIDED BY ENRICO CAMPOREALE "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS 
% BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
% GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
% STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                              
                                              
