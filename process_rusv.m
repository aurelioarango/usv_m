%read files
%pwd
%fprintf(pwd)
%cd 305/
%cd Lab\\Aurelio_Thesis_Folder\\305
%pwd

% Zoom into images, create 64x64 image, from 20khz to 84khz
% from the original images
% decrease the range to capture 22khz calls and test with real data

% Check for number of arguments
function process_rusv(varargin)
%celldisp(varargin);
    % Check if the correct number of inputs is correct
    if(nargin ~= 1)
        fprintf('Usage process <Path To Wave files>\n');
        return;
    else
        main(varargin);
    end
end
%%%
%%% Main Program
%%%
function main(varargin)

curr_dir = pwd;
image_dir = fullfile(curr_dir, 'images')

%cd varargin(1);
path_wave_dir = varargin{1}{1};

%grab all the files inside the directory
wave_files = dir (path_wave_dir);
%wave_files = wave_files(~cellfun('isempty',(regexp(wave_files.name,'\.wav'))))
%remove . and ..
wave_files = wave_files(~ismember({wave_files.name},{'.','..'}));
%numel(wave_files)
%wave_files.name

handles=mupet_initialize();
handles = load_wavfiles(handles, path_wave_dir);
handles.flist = handles.flist.';
handles.image_dir = image_dir;


%create loop to traverse files

%check if dir exist otherwise make one
%%{
%dir_status = exist ('images')
if ~ exist('images', 'dir') 
    fprintf('Creating images directory');
    mkdir ('images');
else
    fprintf('Exists \n');
end

%%}

%MAIN LOOP
%%{
images_dir = handles.image_dir;
for j=1:numel(wave_files)
    %process all files 
    handles=process_file(handles,j);
    %handles
    [rows, elements]=size(handles.syllable_data);
    handles.num_elements = elements;
    %process each syllable

    sub_dir = extractBefore(handles.filename,'.WAV');

    mkdir('images', sub_dir);
    handles.image_dir = fullfile(images_dir, sub_dir);
    
    for i=1:(handles.num_elements)
        show_syllables(handles,i);
    end
end
%%}

%handles
%
%{
%process each file for usv extraction
for i=1:numel(wave_files)
    %only process wave files
    if contains(wave_files(i).name, '.wav')
        %fprintf("%d ",i);
        %compute_usv(dir_path,wavefile);
    end
end
%}
%fprinf(wavefile_1)
% show_syllables
end

%%%
%%% UTILITY FUNCTIONS - MUPET INITIALIZATION
%%%

function handles=mupet_initialize()

    handles.flist='';
    handles.datadir='';
    handles.repertoiredir='repertoires';
    handles.datasetdir='datasets';
    handles.denoising=false;

    % feature extraction
    handles.feature_ext_segment_duration=10; % seconds
    handles.frame_shift_ms=0.0004*4; % ms
    handles.frame_win_ms=0.002; % ms

    % syllable detection
    handles.syllable_stats='';
    handles.sample_frequency=0;
    handles.filename='';
    handles.max_inter_syllable_distance=200;
    handles.psd_smoothing_window_freq=10;

    % repertoire learning
    handles.repertoire_learning_min_nb_syllables_fac=1;
    handles.repertoire_unit_size_seconds=200; % 76.8 for 64k window
    handles.patch_window=handles.repertoire_unit_size_seconds/handles.frame_shift_ms*1e-3; % ms divided by frameshift
    

    % config file
    handles.configdefault{1}=0.0; % noise_reduction_sigma_default
    handles.configdefault{2}=8; % min_syllable_duration_default
    handles.configdefault{3}=handles.repertoire_unit_size_seconds; % max_syllable_duration_default
    handles.configdefault{4}=-95; % min_syllable_total_energy_default
    handles.configdefault{5}=-50; % min_syllable_peak_amplitude_default
    handles.configdefault{6}=5; % min_syllable_distance_default
    handles.configfile=fullfile(pwd,'config.csv');
    handles = create_configfile(handles, true);

end

%%%
%%% UTILITY FUNCTIONS - AUDIO
%%%

% load_wavfiles
function handles=load_wavfiles(handles, path)

    handles.datadir = path;
    handles.wav_directory = path;
    handles.wave_list = [];
    filelist1=dir(fullfile(handles.datadir,'*.WAV'));
    filelist2=dir(fullfile(handles.datadir,'*.wav'));
    crit = '^[^.]+';
    rxResult1 = regexp( {filelist1.name}, crit );
    rxResult2 = regexp( {filelist2.name}, crit );
    
    if (length(filelist1)+length(filelist2)) > 0,
        handles.flist=unique({ filelist1.name filelist2.name });
        
        handles.flist([cellfun(@isempty,rxResult1)==true cellfun(@isempty,rxResult2)==true])=[];
        content=handles.flist;
        [~,handles.audiodir]=fileparts(handles.datadir);
        handles.audiodir=fullfile('audio',handles.audiodir);
        if ~exist(handles.audiodir,'dir')
           mkdir(handles.audiodir);
        end
    else
        sprintf('No wave files found in directory\n');
        %content=sprintf('No wave files found in directory\n');
    end

end

% create_configfile
function handles = create_configfile(handles, flag)

    if ~exist('flag', 'var')
        flag=false;
    end

    if ~exist(handles.configfile,'file') || flag
        configfile=handles.configfile;
        configfile_string=sprintf('%s%s%s', ...
            sprintf('noise-reduction,%.1f\n',handles.configdefault{1}), ...
            sprintf('minimum-syllable-duration,%.1f\n',handles.configdefault{2}), ...
            sprintf('maximum-syllable-duration,%.1f\n',handles.configdefault{3}), ...
            sprintf('minimum-syllable-total-energy,%.1f\n',handles.configdefault{4}), ...
            sprintf('minimum-syllable-peak-amplitude,%.1f\n',handles.configdefault{5}), ...
            sprintf('minimum-syllable-distance,%.1f',handles.configdefault{6}));
        fileID=fopen(configfile,'w');
        fwrite(fileID, configfile_string);
        fclose(fileID);

        % set settings
        handles.config{1}=handles.configdefault{1};
        handles.config{2}=handles.configdefault{2};
        handles.config{3}=handles.configdefault{3};
        handles.config{4}=handles.configdefault{4};
        handles.config{5}=handles.configdefault{5};
        handles.config{6}=handles.configdefault{6};

    end
end


% process_file
function handles=process_file(handles,selected_file)

    %wav_items=get(handles.wav_list,'string');
    wav_items = handles.flist;
    %selected_wav=get(handles.wav_list,'value');
    selected_wav = selected_file;
    wav_dir=handles.wav_directory;
    
    if ~isempty(wav_dir)
        [syllable_data, syllable_stats, filestats, fs]=compute_musv(wav_dir,wav_items(selected_wav),handles);
        handles.syllable_stats = syllable_stats;
        handles.syllable_data = syllable_data;
        handles.sample_frequency = fs;
        handles.filename = wav_items{selected_wav};
        nb_syllables=filestats.nb_of_syllables;
        if nb_syllables >= 1
            
            %syllable_ndx=1;

            % make syllable patch
           % show_syllables(handles,syllable_ndx);
        else
            errordlg(sprintf(' ***              No syllables found in file              *** '),'MUPET info');
        end
    end

end



%%%
%%% UTILITY FUNCTION - LEVEL DOWN
%%%


% compute_musv
function [syllable_data, syllable_stats, filestats, fs] = compute_musv(datadir,flist,handles,existonly)

    % FFT
    Nfft=256*2;

    if ~exist('existonly', 'var')
        existonly=false;
    end

    % Gammatone filterbank
    NbChannels=64;
    %fsMin=90000;
    fsMin=18000;
    fs=250000;
    frame_shift=floor(handles.frame_shift_ms*fs);
    frame_win=floor(handles.frame_win_ms*fs);
    gtdir=fullfile(handles.audiodir);
    if ~exist(gtdir,'dir')
      mkdir(gtdir)
    end
    GTB=gammatone_matrix_sigmoid(Nfft*2,fs,NbChannels);

    % compute FFT of the audio
    cnt = 1;
    init_waitbar = 1;
    for fnameID = 1:length(flist)

         fname=flist{fnameID};
         [~, filename]= fileparts(fname);
         gtfile=fullfile(gtdir, sprintf('%s.mat', filename));

         % info
         fprintf('Processing file %s  ', filename);
         flag = false;

         if exist(gtfile,'file') && ~existonly,
             load(gtfile,'filestats');
             flag = ~isequal(filestats.configpar,handles.config);
         end

         if ~exist(gtfile,'file') || flag

             if init_waitbar==1
                h = waitbar(0,'Initializing processing...');
                init_waitbar=0;
             end
             waitbar(cnt/(2*length(flist)),h,sprintf('Processing files... (file %i of %i)',(cnt-1)/2+1,length(flist)));
             audiofile=fullfile(datadir, fname);

             clear info;
             if exist('audioread')
                 info=audioinfo(audiofile);
             else
                 [~,info.SampleRate]=AUDIOREAD(audiofile);
                 info.TotalSamplesArray=AUDIOREAD(audiofile,'size');
                 info.TotalSamples=info.TotalSamplesArray(1);
             end

             segment_samples=floor(info.SampleRate*handles.feature_ext_segment_duration); % seconds to samples
             nb_of_segments=ceil(info.TotalSamples/segment_samples);
             segment_start_rewind=0;
             frame_start=1;

             resample_normal=true;
             try
                 tmp=resample(audio_segment(1:min(100,info.TotalSamples)),fs,info.SampleRate);
             catch
                 resample_normal=false;
             end

             syllable_data=[];
             for segmentID = 1:nb_of_segments
                 audio_start=(segmentID-1)*segment_samples+1-segment_start_rewind;
                 audio_end=min(segmentID*segment_samples,info.TotalSamples);
                 if exist('audioread')
                    audio_segment=audioread(audiofile,[audio_start audio_end]);
                 else
                    audio_segment=AUDIOREAD(audiofile,[audio_start audio_end]);
                 end
                 if info.SampleRate < fsMin
                     errordlg(sprintf('Sampling rate of audio file %s is too low (<%i kHz). \nPlease delete audio file in directory to proceed.', fname, fsOrig),'Audio file sampling rate too low');
                     continue;
                 end
                 [syllable_data_segment,segment_start_rewind,~,nb_of_frames]=compute_musv_segment(handles,audio_segment,info.SampleRate,GTB,frame_start,audiofile,resample_normal);
                 frame_start=frame_start+nb_of_frames;
                 if ~isempty(syllable_data_segment)
                     syllable_data=[syllable_data syllable_data_segment];
                     %fprintf( "Segment ID %d\n",segmentID );
                 end
             end
             TotNbFrames=floor((info.TotalSamples/info.SampleRate*fs-frame_win+frame_shift)/frame_shift);
             % fprintf( "Total Num Ids %d", TotNbFrames);
             % compute syllable stats for file
             [syllable_data, syllable_stats, filestats] = syllable_activity_file_stats(handles, audiofile, TotNbFrames, syllable_data);
             filestats.configpar=handles.config;

             save(gtfile,'syllable_data','syllable_stats','filestats','-v6');

         else
             if nargout>0
                 load(gtfile,'syllable_data','syllable_stats','filestats');
             end
         end
         fprintf('Done.\n');
         cnt = cnt + 2;
    end
    if exist('h','var')
        close(h);
    end

end

% compute_musv_segment
function [syllable_data,segment_start_rewind,nb_of_syllables,nbFrames] = compute_musv_segment(handles,audio_segment,fsOrig,GTB,frame_start,audiofile,resample_normal)

% parameters
Nfft=512;

% % noise removal
% if handles.denoising
%     min_syllable_duration=5;
%     min_syllable_energy=-1;
% else
%     min_syllable_duration=3;
%     min_syllable_energy=-2;
% end

% energy


logE_thr=0.2;
smooth_fac=floor(5*(0.0016/handles.frame_shift_ms));
smooth_fac_low=10;
grow_fac=floor(3*(0.0016/handles.frame_shift_ms));

% frequency
%fsMin=90000;
fsMin=18000;
fs=250000;
frame_shift=floor(handles.frame_shift_ms*fs);
frame_win=floor(handles.frame_win_ms*fs);

if fsOrig < fsMin
 errordlg(sprintf('Sampling rate of audio file %s is too low (<%i kHz). \nPlease delete audio file in directory to proceed.', fname, fsOrig),'Audio file sampling rate too low');
end
if fsOrig ~= fs
    if resample_normal
        audio_segment=resample(audio_segment,fs,fsOrig);
    else
        T=length(audio_segment)/fsOrig;
        dsfac=fix(sqrt(2^31*fs/fsOrig));
        audio_segment=resample(audio_segment,dsfac,fix(fsOrig/fs*dsfac));
        audio_segment=audio_segment(1:floor(T*fs));
    end
end

% compute MUSV features
%fmin=35000;
fmin=18000;
fmax=110000;
%fmax=90000;
Nmin=floor(Nfft/(fs/2)*fmin);
Nmax=floor(Nfft/(fs/2)*fmax);
[gt_sonogram, sonogram, E_low, E_usv, T]=FE_GT_spectra(audio_segment, fs, frame_win, frame_shift, Nfft, Nmin, Nmax);

size(sonogram);

% Gaussian noise floor estimation by median
logE = log(E_usv);
logE_noise = median(logE);
logE_norm = logE - logE_noise;
logE_low = log(E_low);
logE_low_noise = median(logE_low);
logE_low_norm = logE_low - logE_low_noise;

% syllable activity detection
if handles.denoising
    sad = double(smooth( (logE_norm-logE_low_norm),smooth_fac)>logE_thr);
    sad = double(smooth(sad, grow_fac)>0); % syllable region growing
    sad_low = double(smooth( (logE_low_norm),smooth_fac_low)>logE_thr);
    sad_low = double(smooth(sad_low, grow_fac)>0); % syllable region growing
    sad = (sad-sad_low)>0;
else
    sad = double(smooth(logE_norm,smooth_fac)>logE_thr);
    sad = double(smooth(sad, grow_fac)>0); % syllable region growing
end

start_sad = find((sad(1:end-1)-sad(2:end)) < -.5);
end_sad = find((sad(1:end-1)-sad(2:end)) > .5);
segment_start_rewind=floor((length(audio_segment)-T)/fs*fsOrig);
nbFrames=size(gt_sonogram,2);

%imagesc(gt_sonogram); axis xy; hold on; plot(15*sad,'r'); plot(20*sad_low,'m'); plot(10*((sad-sad_low)>0),'g'); hold off; zoom xon; pause

if ~isempty(end_sad) && ~isempty(start_sad)
    if end_sad(1)<start_sad(1)
        end_sad(1)=[];
    end
    if ~isempty(end_sad)
        se_sad_T=min(length(start_sad),length(end_sad));
        if start_sad(end)>end_sad(end)
           segment_start_rewind=floor((length(audio_segment)-min(T,start_sad(end)*frame_shift))/fs*fsOrig);
           nbFrames=start_sad(end);
        end
        start_sad=start_sad(1:se_sad_T);
        end_sad=end_sad(1:se_sad_T);
    end
else
    start_sad=[];
    end_sad=[];
end

% normalize sonogram by median filtering
noise_floor_GT=median(gt_sonogram,2);
noise_floor_X=median(sonogram,2);
GT = gt_sonogram - repmat(noise_floor_GT, 1, size(gt_sonogram,2));
X = sonogram - repmat(noise_floor_X, 1, size(sonogram,2));
% X=sonogram;

% spectral noise floor
[counts,vals]=hist(GT(:),100);
[sigma_noise, mu_noise] = gaussfit( vals, counts./sum(counts), handles);

% syllable selection
syllable_data=cell(6,length(end_sad));
nb_of_syllables=length(end_sad);
for k=1:nb_of_syllables
    syllable_data{1,k}=audiofile;
    syllable_data{2,k}=GT(:,start_sad(k):end_sad(k)-1); % gt
    syllable_data{3,k}=X(:,start_sad(k):end_sad(k)-1); % fft
    syllable_data{4,k}=E_usv(:,start_sad(k):end_sad(k)-1)*(Nmax-Nmin); % energy
    syllable_data{5,k}=frame_start+start_sad(k)-1;
    syllable_data{6,k}=frame_start+end_sad(k)-1;
    syllable_data{7,k}=1;    % syllable considered for analysis
    syllable_data{8,k}=sigma_noise;
    syllable_data{9,k}=mu_noise;
end

syllable_onset=cell2mat(syllable_data(5,:));
syllable_offset=cell2mat(syllable_data(6,:));
syllable_distance = [syllable_onset(2:end) - syllable_offset(1:end-1) 0];
syllables_to_remove = [];
for k=1:nb_of_syllables
%     fprintf('%.4f %.2f %.2f\n',syllable_data{5,k}*frame_shift/fs, (syllable_data{6,k}-syllable_data{5,k})*frame_shift/fs*1e3 ,syllable_distance(k)*frame_shift/fs*1e3);
    if syllable_distance(k)*frame_shift/fs*1e3 < handles.config{6} && syllable_distance(k)*frame_shift/fs*1e3 > 0
        syllable_data{2,k+1}=GT(:,start_sad(k):end_sad(k+1)-1);
        syllable_data{3,k+1}=X(:,start_sad(k):end_sad(k+1)-1); % fft
        syllable_data{4,k+1}=E_usv(:,start_sad(k):end_sad(k+1)-1)*(Nmax-Nmin); % energy
        syllable_data{4,k+1}=E_usv(:,start_sad(k):end_sad(k+1)-1)*(Nmax-Nmin); % energy
        syllable_data{5,k+1}=syllable_data{5,k};
        syllable_data{6,k+1}=syllable_data{6,k+1};
        syllable_data{7,k+1}=1;    % syllable considered for analysis
        syllable_data{8,k+1}= syllable_data{8,k};
        syllable_data{9,k+1}=syllable_data{9,k};
        syllables_to_remove = [syllables_to_remove k];
    end
end

if ~isempty(syllables_to_remove)
    syllable_data(:,syllables_to_remove)=[];
    nb_of_syllables=size(syllable_data,2);
end

% select syllables
syllable_nbframes=zeros(nb_of_syllables,1);
syllable_energy=zeros(nb_of_syllables,1);
syllable_constant=zeros(nb_of_syllables,1);
for k=1:nb_of_syllables
    syllable_nbframes(k)=size(syllable_data{3,k},2); % nb of syllable frames
    syllable_energy(k)=max(max(log(abs(syllable_data{3,k}+1e-5))));
    syllable_constant(k)=sum(syllable_data{2,k}(:,1)-syllable_data{2,k}(:,end))~=0;
end

% remove constant patches
syllable_selection = syllable_constant ;
syllable_data(:,~syllable_selection)=[];
nb_of_syllables=size(syllable_data,2);

end

% FE_GT_spectra
function [lp,Xf,E_low, E_usv,T]=FE_GT_spectra(sam,fs,FrameLen,FrameShift,Nfft,Nmin,Nmax,W,M)

    GTfloor=1e-3;

    if ~exist('W', 'var')
        W=gammatone_matrix_sigmoid(Nfft*2,fs);
    end
    if ~exist('M', 'var')
        M=1;
    end

    % truncate as in ReadWave
    NbFr=floor( (length(sam)-FrameLen+FrameShift)/FrameShift);
    sam=sam(1:NbFr*FrameShift+FrameLen-FrameShift);

    % % Low pass removal
    % fmax=0.99*fs/2;
    % sam=[0 reshape(sam,1,length(sam))];
    % T=length(sam);

    % Low pass removal
    fmin=18000;
    fmax=0.99*fs/2;
    [b,a] = cheby1(8,  0.5, [fmin fmax]/(fs/2));
    sam=filter(b, a, [0 reshape(sam,1,length(sam))]);
    T=length(sam);

    % framing
    ind1=1:FrameShift:length(sam)-1-FrameLen+FrameShift;

    % new
    win=hamming(FrameLen);
    %win=flattopwin(FrameLen);

    Xf=zeros(Nfft,NbFr);
    for k=1:NbFr
        x=sam(ind1(k):ind1(k)+FrameLen-1)';
        xwin=win.*x;
        X=fft(xwin,2*Nfft);
        Xf(:,k)=X(1:Nfft);
    end
    Xf=abs(Xf).^2;

    % Energy and norm
    E_low=sum(Xf(1:Nmin,:))./Nmin;
    E_usv=sum(Xf(Nmin+1:Nmax,:))./(Nmax-Nmin);

    % GT
    Xf_hp=Xf;
    Xf_hp(1:Nmin,:)=0;
    lp=max(W*Xf_hp,GTfloor);
    lp=log(lp);
    lp=arma_filtering(lp,M);

end

% arma_filtering
function featarma = arma_filtering(featmvn, M)
% ARMA Normalization using M-tap FIR

    featarma = featmvn;
    tmp1 = sum(featarma(:,[1:M]),2);
    tmp2 = sum(featmvn(:,[M+1:2*M+1]),2);
    featarma(:,M+1) = ( tmp1 + tmp2 ) ./ ( 2*M + 1 );
    for t=M+2:size(featmvn,2)-M,
        tmp1 = tmp1 + featarma(:,t-1) - featarma(:,t-M-1);
        tmp2 = tmp2 + featmvn(:,t+M) - featmvn(:,t-1);
        featarma(:,t) = ( tmp1 + tmp2 ) ./ ( 2*M + 1 );
    end

end

% gammatone_matrix_sigmoid
function [wts,cfreqs] = gammatone_matrix_sigmoid(nfft, sr, nfilts, cntfreq, width, maxlen)

    if nargin < 2;    sr = 250000; end
    if nargin < 3;    nfilts = 64; end
    if nargin < 4;    cntfreq = 75000; end
    if nargin < 5;    width = 0.5; end
    if nargin < 6;    maxlen = nfft/2; end

    wts = zeros(nfilts, nfft);

    % ERB filters
    slope = 14.2;
    cfreqs = fix( cntfreq - 1 / (slope * (2/sr)) * log((nfilts+1-[0:nfilts]) ./ [1:nfilts+1] ));

    GTord = 4;

    ucirc = exp(1i*2*pi*[0:(nfft/2)]/nfft);

    for i = 1:nfilts
      cf = cfreqs(i);
      cfn = cfreqs(i+1);
      ERB = width*(cfn-cf);
      B = 1.019*2*pi*ERB;
      r = exp(-B/sr);

      theta = 2*pi*cf/sr;
      pole = r*exp(1i*theta);

      % poles and zeros, following Malcolm's MakeERBFilter
      T = 1/sr;
      A11 = -(2*T*cos(2*cf*pi*T)./exp(B*T) + 2*sqrt(3+2^1.5)*T*sin(2* ...
                                                        cf*pi*T)./exp(B*T))/2;
      A12 = -(2*T*cos(2*cf*pi*T)./exp(B*T) - 2*sqrt(3+2^1.5)*T*sin(2* ...
                                                        cf*pi*T)./exp(B*T))/2;
      A13 = -(2*T*cos(2*cf*pi*T)./exp(B*T) + 2*sqrt(3-2^1.5)*T*sin(2* ...
                                                        cf*pi*T)./exp(B*T))/2;
      A14 = -(2*T*cos(2*cf*pi*T)./exp(B*T) - 2*sqrt(3-2^1.5)*T*sin(2* ...
                                                        cf*pi*T)./exp(B*T))/2;
      zros = -[A11 A12 A13 A14]/T;

      gain(i) =  abs((-2*exp(4*1i*cf*pi*T)*T + ...
                  2*exp(-(B*T) + 2*1i*cf*pi*T).*T.* ...
                  (cos(2*cf*pi*T) - sqrt(3 - 2^(3/2))* ...
                   sin(2*cf*pi*T))) .* ...
                 (-2*exp(4*1i*cf*pi*T)*T + ...
                  2*exp(-(B*T) + 2*1i*cf*pi*T).*T.* ...
                  (cos(2*cf*pi*T) + sqrt(3 - 2^(3/2)) * ...
                   sin(2*cf*pi*T))).* ...
                 (-2*exp(4*1i*cf*pi*T)*T + ...
                  2*exp(-(B*T) + 2*1i*cf*pi*T).*T.* ...
                  (cos(2*cf*pi*T) - ...
                   sqrt(3 + 2^(3/2))*sin(2*cf*pi*T))) .* ...
                 (-2*exp(4*1i*cf*pi*T)*T + 2*exp(-(B*T) + 2*1i*cf*pi*T).*T.* ...
                  (cos(2*cf*pi*T) + sqrt(3 + 2^(3/2))*sin(2*cf*pi*T))) ./ ...
                 (-2 ./ exp(2*B*T) - 2*exp(4*1i*cf*pi*T) +  ...
                  2*(1 + exp(4*1i*cf*pi*T))./exp(B*T)).^4);
      wts(i,1:(nfft/2+1)) = ((T^4)/gain(i)) ...
          * abs(ucirc-zros(1)).*abs(ucirc-zros(2))...
          .*abs(ucirc-zros(3)).*abs(ucirc-zros(4))...
          .*(abs((pole-ucirc).*(pole'-ucirc)).^-GTord);
    end

    wts = wts(:,1:maxlen);
    wts = wts./repmat(max(wts')',1,size(wts,2));

end

% gaussfit
function [sigma, mu] = gaussfit( x, y, handles, sigma0, mu0 )

    % Maximum number of iterations
    Nmax = 5;

    % delete peaks in probability density function
    y=smooth(y,handles.psd_smoothing_window_freq);

    if( length( x ) ~= length( y ))
        fprintf( 'x and y should be of equal length\n\r' );
        exit;
    end

    n = length( x );
    x = reshape( x, n, 1 );
    y = reshape( y, n, 1 );

    %sort according to x
    X = [x,y];
    X = sortrows( X );
    x = X(:,1);
    y = X(:,2);

    %Checking if the data is normalized
    dx = diff( x );
    dy = 0.5*(y(1:length(y)-1) + y(2:length(y)));
    s = sum( dx .* dy );
    if( s > 1.5 || s < 0.5 )
        %fprintf( 'Data is not normalized! The pdf sums to: %f. Normalizing...\n\r', s );
        y = y ./ s;
    end

    X = zeros( n, 3 );
    X(:,1) = 1;
    X(:,2) = x;
    X(:,3) = (x.*x);


    % try to estimate mean mu from the location of the maximum
    [ymax,index]=max(y);
    mu = x(index);

    % estimate sigma
    sigma = 1/(sqrt(2*pi)*ymax);

    if( nargin == 4 )
        sigma = sigma0;
    end

    if( nargin == 5 )
        mu = mu0;
    end

    %xp = linspace( min(x), max(x) );

    % iterations
    for i=1:Nmax
    %    yp = 1/(sqrt(2*pi)*sigma) * exp( -(xp - mu).^2 / (2*sigma^2));
    %    plot( x, y, 'o', xp, yp, '-' );

        dfdsigma = -1/(sqrt(2*pi)*sigma^2)*exp(-((x-mu).^2) / (2*sigma^2));
        dfdsigma = dfdsigma + 1/(sqrt(2*pi)*sigma).*exp(-((x-mu).^2) / (2*sigma^2)).*((x-mu).^2/sigma^3);

        dfdmu = 1/(sqrt(2*pi)*sigma)*exp(-((x-mu).^2)/(2*sigma^2)).*(x-mu)/(sigma^2);

        F = [ dfdsigma dfdmu ];
        a0 = [sigma;mu];
        f0 = 1/(sqrt(2*pi)*sigma).*exp( -(x-mu).^2 /(2*sigma^2));
        a = (F'*F)^(-1)*F'*(y-f0) + a0;
        sigma = a(1);
        mu = a(2);

        if( sigma < 0 )
            sigma = abs( sigma );
            fprintf( 'Instability detected! Rerun with initial values sigma0 and mu0! \n\r' );
            fprintf( 'Check if your data is properly scaled! p.d.f should approx. sum up to \n\r' );
            %exit;
        end
    end

end

% syllable_activity_file_stats
function [syllable_data, syllable_stats, filestats, fs] = syllable_activity_file_stats(handles, audiofile, TotNbFrames, syllable_data, syllable_use, Nfft)

    if ~exist('Nfft', 'var')
        Nfft=512;
    end

    load_data=false;
    if ~exist('syllable_data', 'var')
        load_data=true;
    end

    if ~exist('syllable_use', 'var')
        syllable_use=[];
    end

    GTfloor=1e-1;
    fs=250000;
    frame_shift=floor(handles.frame_shift_ms*fs);

    % load values from config file
    noise_reduction_sigma=handles.config{1};
    min_syllable_duration=handles.config{2};
    max_syllable_duration=handles.config{3};
    min_syllable_total_energy=handles.config{4};
    min_syllable_peak_amplitude=handles.config{5};

    gtbands=gammatone_matrix_sigmoid(Nfft*2,fs);
    gt2fftbands=cell(size(gtbands,1),1);
    for k=1:size(gtbands,1)
        gt2fftbands{k}=find(gtbands(k,:)>0.15);
        % prevent overlapping bands
        if k>1
            [~,fftbands_ndx_remove]=intersect(gt2fftbands{k},gt2fftbands{k-1});
            gt2fftbands{k}(fftbands_ndx_remove)=[];
        end
    end

    % load syllable data
    [~,filename]=fileparts(audiofile);
    if load_data
        gtdir=fullfile(handles.audiodir);
        GT_file=fullfile(gtdir, sprintf('%s.mat', filename));
        if exist(GT_file,'file'),
            load(GT_file,'syllable_data','syllable_stats','filestats');
            if isempty(syllable_data)
                fprintf('File %s not processed.\n', filename);
                return;
            end
            syllable_use=cell2mat(syllable_stats(1,:));
        else
            fprintf('File %s not processed.\n', filename);
            return
        end
    else
        if isempty(syllable_use)
            syllable_use=ones(1,length(syllable_data));
        end
    end
    fprintf('Updating stats of file %s.\n', filename);

    % file stats
    syllable_onset=cell2mat(syllable_data(5,:));
    syllable_offset=cell2mat(syllable_data(6,:));

    % start processing
    syllable_data=syllable_data(:,syllable_use==1);

    nb_of_syllables=size(syllable_data,2);
    syllable_stats=cell(14,nb_of_syllables);
    ndx_remove=zeros(1,nb_of_syllables);

    for k=1:nb_of_syllables

       syllable_stats{7,k}=0; % duration
       syllable_stats{13,k}=0; % duration
       syllable_stats{11,k}=min_syllable_total_energy; % mean energy
       syllable_stats{12,k}=min_syllable_peak_amplitude; % peak energy
       syllable_threshold=log(max(exp(syllable_data{2,k}),GTfloor));
       syllable_threshold=(syllable_threshold-syllable_data{9,k})>noise_reduction_sigma*syllable_data{8,k};

       syllable_threshold_timestamps=find(sum(syllable_threshold));
       if ~isempty(syllable_threshold_timestamps)

           syllable_stats{1,k}=1; % syllable use

           [~,gtmax]=max(syllable_threshold(:,syllable_threshold_timestamps(1)));
           [~,nfftmax]=max(syllable_data{3,k}(gt2fftbands{gtmax},syllable_threshold_timestamps(1)));
           syllable_stats{2,k}=gt2fftbands{gtmax}(nfftmax)/Nfft*fs/2/1e3; % start frequency
           [~,gtmax]=max(syllable_threshold(:,syllable_threshold_timestamps(end)));
           [~,nfftmax]=max(syllable_data{3,k}(gt2fftbands{gtmax},syllable_threshold_timestamps(end)));
           syllable_stats{3,k}=gt2fftbands{gtmax}(nfftmax)/Nfft*fs/2/1e3; % final frequency
           syllable_stats{8,k}=syllable_onset(k)*frame_shift/fs; % start syllable time
           syllable_stats{9,k}=syllable_offset(k)*frame_shift/fs; % end syllable time
           syllable_stats{13,k}=(syllable_stats{9,k} - syllable_stats{8,k})*1e3; % duration before noise reduction

           % mapping syllable_silhouette_gt in syllable_silhouette_fft
           syllable_silhouette_gt = syllable_threshold(:,syllable_threshold_timestamps)>0;
           syllable_stats{7,k}=length(syllable_threshold_timestamps)*frame_shift/fs*1e3; % duration after noise reduction
           gt_occ=sum(syllable_silhouette_gt,2);
           gt_occ_filt=filter([1 1 1]./3,1,gt_occ)>1/3;

           % syllable energy
           syllable_en=[];
           syllable_en_sum=0;
           totnb_fftbins=0;
           for framendx=syllable_threshold_timestamps
               gtbin=find(syllable_threshold(:,framendx)>0);
               fftbins=[];
               for gtbinndx=gtbin
                   fftbins=[fftbins gt2fftbands{gtbinndx}];
               end
               syllable_en_frame=sum(syllable_data{3,k}(fftbins,framendx));
               syllable_en_sum=syllable_en_sum+syllable_en_frame;
               syllable_en=[syllable_en;syllable_data{3,k}(fftbins,framendx)];
               totnb_fftbins=totnb_fftbins+length(fftbins);
           end
           syllable_stats{11,k}=10*log10(max(syllable_en_sum,10^(min_syllable_total_energy/10)));
           syllable_stats{12,k}=10*log10(max(syllable_en));   % peak energy

           % syllable frequency stats
           gt_range=find(gt_occ_filt)';
           if isempty(gt_range)
               ndx_remove(k)=1;
               continue;
           end
           freq_acc=0;
           for gtn=gt_range
               fftsubband=syllable_data{3,k}(gt2fftbands{gtn},syllable_threshold_timestamps);
               maxval_fft_subband=max(max(fftsubband));
               fft_gtval=gt2fftbands{gtn}(sum(fftsubband==maxval_fft_subband,2)==1);
               fft_gtval=fft_gtval(1);
               freq_acc=freq_acc+gt_occ(gtn)*fft_gtval/Nfft*fs/2/1e3;
           end
           syllable_stats{10,k}=freq_acc/sum(gt_occ);
           fftlowerband=syllable_data{3,k}(gt2fftbands{gt_range(1)},syllable_threshold_timestamps);
           fftupperband=syllable_data{3,k}(gt2fftbands{gt_range(end)},syllable_threshold_timestamps);
           maxval_fft_rangemin=max(max(fftlowerband));
           maxval_fft_rangemax=max(max(fftupperband));

           fft_rangemin=gt2fftbands{gt_range(1)}(sum(fftlowerband==maxval_fft_rangemin,2)==1);
           fft_rangemin=fft_rangemin(1);
           fft_rangemax=gt2fftbands{gt_range(end)}(sum(fftupperband==maxval_fft_rangemax,2)==1);
           fft_rangemax=fft_rangemax(1);

           syllable_stats{4,k}=min(syllable_stats{2,k},fft_rangemin/Nfft*fs/2/1e3); % minimum frequency
           %syllable_stats{4,k}=min(gt2fftbands{gt_range(1)}/Nfft*fs/2/1e3);
           syllable_stats{5,k}=max(syllable_stats{3,k},fft_rangemax/Nfft*fs/2/1e3); % maximum frequency
           %syllable_stats{5,k}=min(gt2fftbands{gt_range(end)}/Nfft*fs/2/1e3);
           syllable_stats{6,k}=syllable_stats{5,k}-syllable_stats{4,k}; % frequency bandwidth

       else
           ndx_remove(k)=1;
       end

    end

    if ~isempty(ndx_remove)

        % filtering based on too low total energy
        syllables_total_energy=cell2mat(syllable_stats(11,:));
        ndx_remove=ndx_remove + (syllables_total_energy <= min_syllable_total_energy);

        % filtering based on too low peak energy
        syllables_peak_energy=cell2mat(syllable_stats(12,:));
        ndx_remove=ndx_remove + (syllables_peak_energy <= min_syllable_peak_amplitude);

        % filtering based on too short duration (after noise reduction)
        syllables_duration=cell2mat(syllable_stats(7,:));
        ndx_remove=ndx_remove + (syllables_duration < min_syllable_duration);

        % filtering based on too long duration (after noise reduction)
        ndx_remove=ndx_remove + (syllables_duration > max_syllable_duration);
    end

    %     % filtering based on too short duration
    %     syllables_duration=cell2mat(syllable_stats(13,:));
    %     ndx_remove=ndx_remove + (syllables_duration <= min_syllable_duration);

    % apply syllable removal
    syllable_stats(:,ndx_remove>0)=[];
    syllable_data(:,ndx_remove>0)=[];
    syllable_onset(ndx_remove>0)=[];
    syllable_offset(ndx_remove>0)=[];

    % inter syllable distance computation
    nb_of_syllables=size(syllable_data,2);
    syllable_distance=[syllable_onset(2:end) - syllable_offset(1:end-1) 0];
    for k=1:nb_of_syllables
        if k==nb_of_syllables
            syllable_stats{14,k}=-100; % distance to next syllable
        else
            syllable_stats{14,k}=syllable_distance(k)*frame_shift/fs*1e3; % distance to next syllable
        end
    end

    % filestats
    filestats.TotNbFrames=TotNbFrames;
    filestats.syllable_dur='';
    filestats.syllable_activity='';
    filestats.syllable_distance='';
    filestats.nb_of_syllables=0;
    filestats.syllable_count_per_minute='';
    filestats.syllable_count_per_second='';
    if ~isempty(syllable_stats)

        % file statistics
        filestats.syllable_dur = cell2mat(syllable_stats(13,:));
        filestats.syllable_activity = sum(filestats.syllable_dur)/filestats.TotNbFrames;
        filestats.syllable_distance = cell2mat(syllable_stats(14,:));
        filestats.nb_of_syllables = length(filestats.syllable_dur);

        % syllable starting times
        syllable_onset_frames=zeros(1,ceil(syllable_stats{8,end}./frame_shift*fs));
        syllable_onset_frames(floor(cell2mat(syllable_stats(8,:))./frame_shift*fs))=1;

        % minute
        framewin=60*floor(fs/frame_shift);
        ind1=1:framewin:length(syllable_onset_frames)-framewin;
        ind2=(1:framewin)';
        syllable_split_per_minute=sum(syllable_onset_frames(ind1(ones(framewin,1),:)+ind2(:,ones(1,length(ind1)))),1);
        filestats.syllable_count_per_minute = syllable_split_per_minute';

        % second
        framewin=1*floor(fs/frame_shift);
        ind1=1:framewin:length(syllable_onset_frames)-framewin-floor(framewin/2);
        ind2=(1:framewin)';
        syllable_split_per_second=sum(syllable_onset_frames(ind1(ones(framewin,1),:)+ind2(:,ones(1,length(ind1)))),1);
        filestats.syllable_count_per_second = syllable_split_per_second';

    end

end

% show_syllables
function show_syllables(handles,syllable_ndx)

    
    % make syllable patch
    syllable_gt = handles.syllable_data{2,syllable_ndx};
    syllable_duration=size(syllable_gt,2);
    syllable_patch_window=max(handles.patch_window,ceil(syllable_duration/2)*2);
    %syllable_patch_gt = ones(size(syllable_gt,1), syllable_patch_window);
    syllable_patch_gt = zeros(size(syllable_gt,1), syllable_patch_window);
    syllable_patch_window_start=floor(syllable_patch_window/2)-floor(syllable_duration/2);
    syllable_patch_gt(:, syllable_patch_window_start+1:syllable_patch_window_start+syllable_duration) = syllable_gt;
    syllable_fft = handles.syllable_data{3,syllable_ndx};

    syllable_fft_median=median(syllable_fft(:));
    syllable_fft_median=2*syllable_fft_median;
    %random_num = 0.0000017;
    %random_num = -15.5;
    %syllable_patch_fft = syllable_fft_median*ones(size(syllable_fft,1), syllable_patch_window);
    
    syllable_patch_fft = syllable_fft_median * rand(size(syllable_fft,1), syllable_patch_window);
    
    %syllable_patch_fft = random_num * syllable_patch_fft;%Adding noise
    
    syllable_duration=size(syllable_fft,2);
    
    syllable_patch_fft(:, syllable_patch_window_start+1:syllable_patch_window_start+syllable_duration) = syllable_fft;
    
    %syllable_patch_window_start; % this is the position start of the call
    %syllable_duration; % this is the call duration
    
    %syllable_fft_median
    %_le
    %fprintf("processing Images \n ");
    % fft figure
    %axes(handles.syllable_axes_fft);
    syllable_patch_fft_dB=10*log10(abs(syllable_patch_fft(1:2:end,:)+1e-5)); % in dB
    %syllable_patch_fft
    
    %fft_range_db1=-95;
    
    fft_range_db1_min=-50;
    fft_range_db2=0;
    fft_peak_db=handles.syllable_stats{12,syllable_ndx};
    fft_range=[fft_range_db1_min , fft_peak_db+fft_range_db2];
    %fft_peak_db
    %syllable_patch_fft_dB; % image 
    %fft_range
    
    imagesc(syllable_patch_fft_dB,fft_range); 
    axis xy; 
    colorbar;


    colormap jet; 
    %colormap hot;
 
     %----------writing syllable to png file ------------%

    [pathstr, name, ext] = fileparts(handles.filename);
    img_filename = sprintf('%s_%d.png',name, syllable_ndx);

  


    %-----------end of writing image to file ------------%

    %size(syllable_patch_fft_dB,1)/5:size(syllable_patch_fft_dB,1);
    set(gca,'YTick',[0:size(syllable_patch_fft_dB,1)/5:size(syllable_patch_fft_dB,1)]) % FFT bands
    set(gca,'YTickLabel',fix([0:handles.sample_frequency/2/5:handles.sample_frequency/2]/1e3)) % FFT bands
    %set(gca,'XTick',[0:syllable_patch_window/6:syllable_patch_window]) % frequency
    %set(gca, 'XTickLabel', fix([64]));
    %set(gca,'XTickLabel',fix([0:handles.frame_shift_ms*syllable_patch_window/6:syllable_patch_window*handles.frame_shift_ms]*1e3)) % frequency
    set(gca, 'FontSize',11,'FontName','default');
    %xlabel('Time (milliseconds)','FontSize',11,'FontName','default');
    ylabel('Frequency [kHz]','FontSize',11,'FontName','default');
    %title('Sonogram','FontSize',11,'FontName','default','FontWeight','bold');
    ylim([size(syllable_fft,1)/125000*25000/2 size(syllable_fft,1)/2]);
    %ylim([size(syllable_fft,1)/100000*30000/2 size(syllable_fft,1)/2]);
    %ylim([0 250]);
    
%{ 
syll_window = 0;
    
    if syllable_duration  < 64
        syll_window = 64 - 5;
    elseif syllable_duration < 128 && syllable_duration > 64 
        syll_window = 128 - 5;
    elseif syllable_duration < 256 && syllable_duration > 128
        syll_window = 256 - 5;
    elseif syllable_duration < 512 && syllable_duration > 256
        syll_window = 512 - 5;
            
    end
    syll_window
    
    xlim([syllable_patch_window_start-5 syllable_patch_window_start+syll_window]);%size of 64 x-axis length
%}
    
    img = getframe(gca);
    [usv map_usv] = frame2im(img);
    %img_size = size(img.cdata)
    %usv = imsharpen(usv);
    usv = imresize(usv,[512 512]);
    %handles
    filename = fullfile(handles.image_dir,img_filename );
    imwrite(usv,filename,'png');
  
end


% getCDataHandles
function hout = getCDataHandles(h)
    % getCDataHandles  Find all objects with indexed CData

    %recursively descend object tree, finding objects with indexed CData
    % An exception: don't include children of objects that themselves have CData:
    %   for example, scattergroups are non-standard hggroups, with CData. Changing
    %   such a group's CData automatically changes the CData of its children,
    %   (as well as the children's handles), so there's no need to act on them.

    error(nargchk(1,1,nargin,'struct'))

    hout = [];
    if isempty(h),return;end

    ch = get(h,'children');
    for hh = ch'
        g = get(hh);
        if isfield(g,'CData'),     %does object have CData?
            %is it indexed/scaled?
            if ~isempty(g.CData) && isnumeric(g.CData) && size(g.CData,3)==1,
                hout = [hout; hh]; %#ok<AGROW> %yes, add to list
            end
        else %no CData, see if object has any interesting children
                hout = [hout; getCDataHandles(hh)]; %#ok<AGROW>
        end
    end

end

% getParentAxes
function hAx = getParentAxes(h)
% getParentAxes  Return enclosing axes of a given object (could be self)

    error(nargchk(1,1,nargin,'struct'))
    %object itself may be an axis
    if strcmp(get(h,'type'),'axes'),
        hAx = h;
        return
    end

    parent = get(h,'parent');
    if (strcmp(get(parent,'type'), 'axes')),
        hAx = parent;
    else
        hAx = getParentAxes(parent);
    end

end


% cbfreeze
function CBH = cbfreeze(varargin)


    % INPUTS CHECK-IN
    % -------------------------------------------------------------------------

    % Parameters:
    appName = 'cbfreeze';

    % Set defaults:
    OPT  = 'on';
    H    = get(get(0,'CurrentFigure'),'CurrentAxes');
    CMAP = [];

    % Checks inputs:
    assert(nargin<=3,'CAVARGAS:cbfreeze:IncorrectInputsNumber',...
        'At most 3 inputs are allowed.')
    assert(nargout<=1,'CAVARGAS:cbfreeze:IncorrectOutputsNumber',...
        'Only 1 output is allowed.')

    % Checks from where CBFREEZE was called:
    if (nargin~=2) || (isempty(varargin{1}) || ...
            ~all(reshape(ishandle(varargin{1}),[],1)) ...
            || ~isempty(varargin{2}))
        % CBFREEZE called from Command Window or M-file:

        % Reads H in the first input: Version 2.1
        if ~isempty(varargin) && ~isempty(varargin{1}) && ...
                all(reshape(ishandle(varargin{1}),[],1))
            H = varargin{1};
            varargin(1) = [];
        end

        % Reads CMAP in the first input: Version 2.1
        if ~isempty(varargin) && ~isempty(varargin{1})
            if isnumeric(varargin{1}) && (size(varargin{1},2)==3) && ...
                    (size(varargin{1},1)==numel(varargin{1})/3)
                CMAP = varargin{1};
                varargin(1) = [];
            elseif ischar(varargin{1}) && ...
                    (size(varargin{1},2)==numel(varargin{1}))
                temp = figure('Visible','off');
                try
                    CMAP = colormap(temp,varargin{1});
                catch
                    close temp
                    error('CAVARGAS:cbfreeze:IncorrectInput',...
                        'Incorrrect colormap name ''%s''.',varargin{1})
                end
                close temp
                varargin(1) = [];
            end
        end

        % Reads options: Version 2.1
        while ~isempty(varargin)
            if isempty(varargin{1}) || ~ischar(varargin{1}) || ...
                    (numel(varargin{1})~=size(varargin{1},2))
                varargin(1) = [];
                continue
            end
            switch lower(varargin{1})
                case {'off','of','unfreeze','unfreez','unfree','unfre', ...
                        'unfr','unf','un','u'}
                    OPT = 'off';
                case {'delete','delet','dele','del','de','d'}
                    OPT = 'delete';
                otherwise
                    OPT = 'on';
            end
        end

        % Gets colorbar handles or creates them:
        CBH = cbhandle(H,'force');

    else

        % Check for CallBacks functionalities:
        % ------------------------------------

        varargin{1} = double(varargin{1});

        if strcmp(get(varargin{1},'BeingDelete'),'on')
            % CBFREEZE called from DeletFcn:

            if (ishandle(get(varargin{1},'Parent')) && ...
                    ~strcmpi(get(get(varargin{1},'Parent'),'BeingDeleted'),'on'))
                % The handle input is being deleted so do the colorbar:
                OPT = 'delete';

                if strcmp(get(varargin{1},'Tag'),'Colorbar')
                    % The frozen colorbar is being deleted:
                    H = varargin{1};
                else
                    % The peer axes is being deleted:
                    H = ancestor(varargin{1},{'figure','uipanel'});
                end
            else
                % The figure is getting close:
                return
            end

        elseif ((gca==varargin{1}) && ...
                (gcf==ancestor(varargin{1},{'figure','uipanel'})))
            % CBFREEZE called from ButtonDownFcn:

            cbdata = getappdata(varargin{1},appName);
            if ~isempty(cbdata)
                if ishandle(cbdata.peerHandle)
                    % Sets the peer axes as current (ignores mouse click-over):
                    set(gcf,'CurrentAxes',cbdata.peerHandle);
                    return
                end
            else
                % Clears application data:
                rmappdata(varargin{1},appName)
            end
            H = varargin{1};
        end

        % Gets out:
        CBH = cbhandle(H);

    end

    % -------------------------------------------------------------------------
    % MAIN
    % -------------------------------------------------------------------------

    % Keeps current figure:
    cfh = get(0,'CurrentFigure');

    % Works on every colorbar:
    for icb = 1:length(CBH)

        % Colorbar handle:
        cbh = double(CBH(icb));

        % This application data:
        cbdata = getappdata(cbh,appName);

        % Gets peer axes handle:
        if ~isempty(cbdata)
            peer = cbdata.peerHandle;
            if ~ishandle(peer)
                rmappdata(cbh,appName)
                continue
            end
        else
            % No matters, get them below:
            peer = [];
        end

        % Choose functionality:
        switch OPT

            case 'delete'
                % Deletes:
                if ~isempty(peer)
                    % Returns axes to previous size:
                    oldunits = get(peer,'Units');
                    set(peer,'Units','Normalized');
                    set(peer,'Position',cbdata.peerPosition)
                    set(peer,'Units',oldunits)
                    set(peer,'DeleteFcn','')
                    if isappdata(peer,appName)
                        rmappdata(peer,appName)
                    end
                end
                if strcmp(get(cbh,'BeingDelete'),'off')
                    delete(cbh)
                end

            case 'off'
                % Unfrozes:
                if ~isempty(peer)
                    delete(cbh);
                    set(peer,'DeleteFcn','')
                    if isappdata(peer,appName)
                        rmappdata(peer,appName)
                    end
                    oldunits = get(peer,'Units');
                    set(peer,'Units','Normalized')
                    set(peer,'Position',cbdata.peerPosition)
                    set(peer,'Units',oldunits)
                    CBH(icb) = colorbar(...
                        'Peer'    ,peer,...
                        'Location',cbdata.cbLocation);
                end

            case 'on'
                % Freezes:

                % Gets colorbar axes properties:
                cbprops = get(cbh);

                % Current axes on colorbar figure:
                fig = ancestor(cbh,{'figure','uipanel'});
                cah = get(fig,'CurrentAxes');

                % Gets colorbar image handle. Fixed BUG, Sep 2009
                himage = findobj(cbh,'Type','image');

                % Gets image data and transforms them to RGB:
                CData = get(himage,'CData');
                if size(CData,3)~=1
                    % It's already frozen:
                    continue
                end

                % Gets image tag:
                imageTag = get(himage,'Tag');

                % Deletes previous colorbar preserving peer axes position:
                if isempty(peer)
                    peer = cbhandle(cbh,'peer');
                end
                oldunits = get(peer,'Units');
                set(peer,'Units','Normalized')
                position = get(peer,'Position');
                delete(cbh)
                oldposition = get(peer,'Position');

                % Seves axes position
                cbdata.peerPosition = oldposition;
                set(peer,'Position',position)
                set(peer,'Units',oldunits)

                % Generates a new colorbar axes:
                % NOTE: this is needed because each time COLORMAP or CAXIS
                %       is used, MATLAB generates a new COLORBAR! This
                %       eliminates that behaviour and is the central point
                %       on this function.
                cbh = axes(...
                    'Parent'  ,cbprops.Parent,...
                    'Units'   ,'Normalized',...
                    'Position',cbprops.Position...
                    );

                % Saves location for future calls:
                cbdata.cbLocation = cbprops.Location;

                % Move ticks because IMAGE draws centered pixels:
                XLim = cbprops.XLim;
                YLim = cbprops.YLim;
                if     isempty(cbprops.XTick)
                    % Vertical:
                    X = XLim(1) + diff(XLim)/2;
                    Y = YLim    + diff(YLim)/(2*length(CData))*[+1 -1];
                else % isempty(YTick)
                    % Horizontal:
                    Y = YLim(1) + diff(YLim)/2;
                    X = XLim    + diff(XLim)/(2*length(CData))*[+1 -1];
                end

                % Gets colormap:
                if isempty(CMAP)
                    cmap = colormap(fig);
                else
                    cmap = CMAP;
                end

                % Draws a new RGB image:
                image(X,Y,ind2rgb(CData,cmap),...
                    'Parent'            ,cbh,...
                    'HitTest'           ,'off',...
                    'Interruptible'     ,'off',...
                    'SelectionHighlight','off',...
                    'Tag'               ,imageTag)

                % Moves all '...Mode' properties at the end of the structure,
                % so they won't become 'manual':
                % Bug found by Rafa at the FEx. Thanks!, which also solves the
                % bug found by Jenny at the FEx too. Version 2.0
                cbfields = fieldnames(cbprops);
                indmode  = strfind(cbfields,'Mode');
                temp     = repmat({'' []},length(indmode),1);
                cont     = 0;
                for k = 1:length(indmode)
                    % Removes the '...Mode' properties:
                    if ~isempty(indmode{k})
                        cont = cont+1;
                        temp{cont,1} = cbfields{k};
                        temp{cont,2} = getfield(cbprops,cbfields{k});
                        cbprops = rmfield(cbprops,cbfields{k});
                    end
                end
                for k=1:cont
                    % Now adds them at the end:
                    cbprops = setfield(cbprops,temp{k,1},temp{k,2});
                end

                % Removes special COLORBARs properties:
                cbprops = rmfield(cbprops,{...
                    'CurrentPoint','TightInset','BeingDeleted','Type',...       % read-only
                    'Title','XLabel','YLabel','ZLabel','Parent','Children',...  % handles
                    'UIContextMenu','Location',...                              % colorbars
                    'ButtonDownFcn','DeleteFcn',...                             % callbacks
                    'CameraPosition','CameraTarget','CameraUpVector', ...
                    'CameraViewAngle',...
                    'PlotBoxAspectRatio','DataAspectRatio','Position',...
                    'XLim','YLim','ZLim'});

                % And now, set new axes properties almost equal to the unfrozen
                % colorbar:
                set(cbh,cbprops)

                % CallBack features:
                set(cbh,...
                    'ActivePositionProperty','position',...
                    'ButtonDownFcn'         ,@cbfreeze,...  % mhh...
                    'DeleteFcn'             ,@cbfreeze)     % again
                set(peer,'DeleteFcn'        ,@cbfreeze)     % and again!

                % Do not zoom or pan or rotate:
                %if isAllowAxesZoom(fig,cbh)
                setAllowAxesZoom  (    zoom(fig),cbh,false)
                %end
                %if isAllowAxesPan(fig,cbh)
                setAllowAxesPan   (     pan(fig),cbh,false)
                %end
                %if isAllowAxesRotate(fig,cbh)
                setAllowAxesRotate(rotate3d(fig),cbh,false)
                %end

                % Updates data:
                CBH(icb) = cbh;

                % Saves data for future undo:
                cbdata.peerHandle = peer;
                cbdata.cbHandle   = cbh;
                setappdata(cbh ,appName,cbdata);
                setappdata(peer,appName,cbdata);

                % Returns current axes:
                if ishandle(cah)
                    set(fig,'CurrentAxes',cah)
                end

        end % switch functionality

    end  % MAIN loop

    % Resets the current figure
    if ishandle(cfh)
        set(0,'CurrentFigure',cfh)
    end

    % OUTPUTS CHECK-OUT
    % -------------------------------------------------------------------------

    % Output?:
    if ~nargout
        clear CBH
    else
        CBH(~ishandle(CBH)) = [];
    end

end

% cbhandle
function CBH = cbhandle(varargin)

% INPUTS CHECK-IN
% -------------------------------------------------------------------------

    % Parameters:
    appName = 'cbfreeze';

    % Sets default:
    H      = get(get(0,'CurrentFigure'),'CurrentAxes');
    FORCE  = false;
    UNHIDE = false;
    PEER   = false;

    % Checks inputs/outputs:
    assert(nargin<=5,'CAVARGAS:cbhandle:IncorrectInputsNumber',...
        'At most 5 inputs are allowed.')
    assert(nargout<=1,'CAVARGAS:cbhandle:IncorrectOutputsNumber',...
        'Only 1 output is allowed.')

    % Gets H: Version 2.1
    if ~isempty(varargin) && ~isempty(varargin{1}) && ...
            all(reshape(ishandle(varargin{1}),[],1))
        H = varargin{1};
        varargin(1) = [];
    end

    % Gets UNHIDE:
    while ~isempty(varargin)
        if isempty(varargin) || isempty(varargin{1}) || ~ischar(varargin{1})...
                || (numel(varargin{1})~=size(varargin{1},2))
            varargin(1) = [];
            continue
        end

        switch lower(varargin{1})
            case {'force','forc','for','fo','f'}
                FORCE  = true;
            case {'unhide','unhid','unhi','unh','un','u'}
                UNHIDE = true;
            case {'peer','pee','pe','p'}
                PEER   = true;
        end
        varargin(1) = [];
    end

    % -------------------------------------------------------------------------
    % MAIN
    % -------------------------------------------------------------------------

    % Show hidden handles:
    if UNHIDE
        UNHIDE = strcmp(get(0,'ShowHiddenHandles'),'off');
        set(0,'ShowHiddenHandles','on')
    end

    % Forces colormaps
    if isempty(H) && FORCE
        H = gca;
    end
    H = H(:);
    nH = length(H);

    % Checks H type:
    newH = [];
    for cH = 1:nH
        switch get(H(cH),'type')

            case {'figure','uipanel'}
                % Changes parents to its children axes
                haxes = findobj(H(cH), '-depth',1, 'Type','Axes', ...
                    '-not', 'Tag','legend');
                if isempty(haxes) && FORCE
                    haxes = axes('Parent',H(cH));
                end
                newH = [newH; haxes(:)];

            case 'axes'
                % Continues
                newH = [newH; H(cH)];
        end

    end
    H  = newH;
    nH = length(H);

    % Looks for CBH on axes:
    CBH = NaN(size(H));
    for cH = 1:nH

        % If its peer axes then one of the following is not empty:
        hin  = double(getappdata(H(cH),'LegendColorbarInnerList'));
        hout = double(getappdata(H(cH),'LegendColorbarOuterList'));

        if ~isempty([hin hout]) && any(ishandle([hin hout]))
            % Peer axes:

            if ~PEER
                if ishandle(hin)
                    CBH(cH) = hin;
                else
                    CBH(cH) = hout;
                end
            else
                CBH(cH) = H(cH);
            end

        else
            % Not peer axes:

            if isappdata(H(cH),appName)
                % CBFREEZE axes:

                appdata = getappdata(H(cH),appName);
                if ~PEER
                    CBH(cH) = double(appdata.cbHandle);
                else
                    CBH(cH) = double(appdata.peerHandle);
                end

            elseif strcmp(get(H(cH),'Tag'),'Colorbar')
                % Colorbar axes:

                if ~PEER

                    % Saves its handle:
                    CBH(cH) = H(cH);

                else

                    % Looks for its peer axes:
                    peer = findobj(ancestor(H(cH),{'figure','uipanel'}), ...
                        '-depth',1, 'Type','Axes', ...
                        '-not', 'Tag','Colorbar', '-not', 'Tag','legend');
                    for l = 1:length(peer)
                        hin  = double(getappdata(peer(l), ...
                            'LegendColorbarInnerList'));
                        hout = double(getappdata(peer(l), ...
                            'LegendColorbarOuterList'));
                        if any(H(cH)==[hin hout])
                            CBH(cH) = peer(l);
                            break
                        end
                    end

                end

            else
                % Just some normal axes:

                if FORCE
                    temp = colorbar('Peer',H(cH));
                    if ~PEER
                        CBH(cH) = temp;
                    else
                        CBH(cH) = H(cH);
                    end
                end
            end

        end

    end

    % Hidden:
    if UNHIDE
        set(0,'ShowHiddenHandles','off')
    end

    % Clears output:
    CBH(~ishandle(CBH)) = [];

% % %
end

