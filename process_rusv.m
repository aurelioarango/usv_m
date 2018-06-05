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
        fprintf("Usage process <Path To Wave files>\n");
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
    handles.configdefault{1}=0; % noise_reduction_sigma_default
    handles.configdefault{2}=8; % min_syllable_duration_default
    handles.configdefault{3}=handles.repertoire_unit_size_seconds; % max_syllable_duration_default
    handles.configdefault{4}=-65; % min_syllable_total_energy_default
    handles.configdefault{5}=-25; % min_syllable_peak_amplitude_default
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

% default_configfile
function handles = default_configfile(handles)
    create_configfile(handles,true);
    msgbox(sprintf(' ***  Settings changed to default values. ***'),'MUPET info');
end

% load_configfile
function handles = load_configfile(handles)

    configfile=handles.configfile;
    try
        [parameterfield,vals]=textread(configfile,'%s%f','delimiter',',');

        noise_reduction_ndx=cellfun(@isempty,strfind(parameterfield,'noise-reduction'))==0;
        min_syllable_duration_ndx=cellfun(@isempty,strfind(parameterfield,'minimum-syllable-duration'))==0;
        max_syllable_duration_ndx=cellfun(@isempty,strfind(parameterfield,'maximum-syllable-duration'))==0;
        min_syllable_total_energy_ndx=cellfun(@isempty,strfind(parameterfield,'minimum-syllable-total-energy'))==0;
        min_syllable_peak_amplitude_ndx=cellfun(@isempty,strfind(parameterfield,'minimum-syllable-peak-amplitude'))==0;
        min_syllable_distance_ndx=cellfun(@isempty,strfind(parameterfield,'minimum-syllable-distance'))==0;

        handles.config{1} = vals(noise_reduction_ndx);
        if (handles.config{1}<0) || (handles.config{1}>10)
            ME = MException('MyComponent:noSuchVariable','Verify noise reduction parameter');
            throw(ME);
        end
        handles.config{2} = vals(min_syllable_duration_ndx);
        if (handles.config{2}<0)
            ME = MException('MyComponent:noSuchVariable','Verify minimum syllable duration parameter');
            throw(ME);
        end
        handles.config{3} = vals(max_syllable_duration_ndx);
        if (handles.config{3}>handles.repertoire_unit_size_seconds)
            ME = MException('MyComponent:noSuchVariable','Verify maximum syllable duration parameter');
            throw(ME);
        end
        handles.config{4}= vals(min_syllable_total_energy_ndx);
        handles.config{5} = vals(min_syllable_peak_amplitude_ndx);
        handles.config{6} = vals(min_syllable_distance_ndx);
        if (handles.config{6}<0)
            ME = MException('MyComponent:noSuchVariable','Verify syllable distance parameter');
            throw(ME);
        end
        msgbox(sprintf(' ***  Settings successfully loaded ***'),'MUPET info');
    catch ME
        if strcmp(ME.identifier,'MATLAB:dataread:TroubleReading')
            msgbox(sprintf(' *** Incorrect format of config file. Using default settings ***'),'MUPET info');
        else
            msgbox(sprintf(' ***  %s Using default settings ***',ME.message),'MUPET info');
        end
        handles.config{1}=handles.configdefault{1};
        handles.config{2}=handles.configdefault{2};
        handles.config{3}=handles.configdefault{3};
        handles.config{4}=handles.configdefault{4};
        handles.config{5}=handles.configdefault{5};
        handles.config{6}=handles.configdefault{6};
        create_configfile(handles);
    end

end

% noise_reduction
function handles=noise_reduction(hObject, handles)

    button_state = get(hObject,'Value');
    if button_state == get(hObject,'Max')
        set(hObject,'ForegroundColor',[0 .75 0.35]);
        set(hObject,'String','denoise ON');
        handles.denoising=true;
        handles.origaudiodir=handles.audiodir;
        handles.audiodir=sprintf('%s_denoiseON',handles.audiodir);
    elseif button_state == get(hObject,'Min')
        set(hObject,'ForegroundColor',[0 0 0]);
        set(hObject,'String','denoise OFF');
        handles.denoising=false;
        handles.audiodir=handles.origaudiodir;
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
            
            syllable_ndx=1;

            % make syllable patch
           % show_syllables(handles,syllable_ndx);
        else
            errordlg(sprintf(' ***              No syllables found in file              *** '),'MUPET info');
        end
    end

end

% ignore_file
function handles=ignore_file(handles)

    wav_items=get(handles.wav_list,'string');
    selected_wav=get(handles.wav_list,'value');
    if ~isempty(wav_items)
        handles.flist(strcmp(handles.flist,wav_items{selected_wav}))=[];
        wav_items(selected_wav)=[];
        set(handles.wav_list,'value',1);
        set(handles.wav_list,'string',wav_items);
    end
end

% process_all_files
function handles=process_all_files(handles)

    %wav_items=get(handles.wav_list,'string');
    %get the list of items
    wav_items= handles.flist;
    
    %wav_dir=get(handles.wav_directory,'string');
    wav_dir = handles.wav_directory;
    if ~isempty(wav_dir)
        compute_musv(wav_dir,wav_items,handles);
    end
    msgbox(sprintf(' ***              All files processed              *** '),'MUPET info');

end

% file_report
function handles=file_report(handles)

    wav_items=get(handles.wav_list,'string');
    wav_dir=get(handles.wav_directory,'string');
    if ~isempty(wav_dir)
        compute_csv_stats(wav_dir,wav_items,handles);
    end
    msgbox(sprintf('***              All files exported to CSV files              ***\n See folder %s/CSV',handles.audiodir),'MUPET info');

end

% process_restart
function process_restart(handles)

    wav_items=get(handles.wav_list,'string');
    for k=1:length(wav_items)
        [~,filename]=fileparts(wav_items{k});
        syllable_file=fullfile(handles.audiodir, sprintf('%s.mat', filename));
        if exist(syllable_file,'file'),
            delete(syllable_file);
        end
    end
    msgbox(sprintf(' ***            Processing restart succeeded.            *** '),'MUPET info');

end


%%%
%%% UTILITY FUNCTIONS - REPERTOIRES
%%%

% set_nbUnits
function set_nbUnits(handles)
    units=get(handles.nbunits,'String');
    NbUnits=str2double(units(get(handles.nbunits,'Value')));
    repertoire_content=dir(fullfile(handles.repertoiredir,sprintf('*_N%i.mat',NbUnits)));
    repertoire_refined_content=dir(fullfile(handles.repertoiredir,sprintf('*_N%i*+.mat',NbUnits)));
    repertoire_content(end+1:end+length(repertoire_refined_content))=repertoire_refined_content;
    categoriesel=cellfun(@num2str,mat2cell([5:5:length(repertoire_content)*NbUnits]',ones(length(repertoire_content)*NbUnits/5,1)),'un',0);
    if ~isempty(categoriesel)
        set(handles.categories,'string',categoriesel);
    end
    set(handles.repertoire_list,'value',1);
    set(handles.repertoire_list,'string',{repertoire_content.name});
    set(handles.selected_repertoire_A,'string','');
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
    %win=hamming(FrameLen);
    win=flattopwin(FrameLen);

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

% syllable_activity_stats
function syllable_activity_stats(handles, wavefile_list, dataset_filename, Nfft)

    if ~exist('Nfft', 'var')
        Nfft=512;
    end
    fs=250000;
    frame_shift=floor(handles.frame_shift_ms*fs);

    gtdir=fullfile(handles.audiodir);

    % Accumulate GT sonogram frames
    dataset_stats.syllable_dur=[];
    dataset_stats.syllable_distance=[];
    dataset_stats.syllable_activity=[];
    dataset_stats.syllable_count_per_minute=[];
    dataset_stats.syllable_count_per_second=[];
    dataset_stats.file_length=[];
    dataset_stats.filenames=[];
    dataset_stats.length=0;
    Xn_frames=0; Xn_tot=0;
    nb_of_syllables=0;
    dataset_content={};

    for wavefileID = 1:length(wavefile_list)
        [~, wavefile]= fileparts(wavefile_list{wavefileID});
        syllable_file=fullfile(gtdir, sprintf('%s.mat', wavefile));
        if exist(syllable_file,'file'),
            % info
            fprintf('Processing file %s', wavefile);
            load(syllable_file,'syllable_data','filestats');
            syllable_data(2,:)={[]};
            dataset_content{end+1}=sprintf('%s.mat', wavefile);

            if filestats.nb_of_syllables >=1

                % accumulate syllable stats
                dataset_stats.syllable_dur = [dataset_stats.syllable_dur filestats.syllable_dur];
                dataset_stats.syllable_distance = [dataset_stats.syllable_distance filestats.syllable_distance];
                dataset_stats.file_length = [dataset_stats.file_length; filestats.syllable_activity*filestats.TotNbFrames];
                dataset_stats.syllable_count_per_minute = [dataset_stats.syllable_count_per_minute; filestats.syllable_count_per_minute ];
                dataset_stats.syllable_count_per_second = [dataset_stats.syllable_count_per_second; filestats.syllable_count_per_second ];
                dataset_stats.length = dataset_stats.length + filestats.TotNbFrames;
                dataset_stats.filenames = [dataset_stats.filenames syllable_data(1,:)];

                % accumulate psd
                for syllableID = 1:filestats.nb_of_syllables
                    E=cell2mat(syllable_data(4,syllableID));
                    E(E==0)=1;
                    sumXn=sum(cell2mat(syllable_data(3,syllableID))./(ones(Nfft,1)*E),2);
                    Xn_tot = Xn_tot + sumXn;
                    Xn_frames=Xn_frames+length(E);
                end
                nb_of_syllables=nb_of_syllables+filestats.nb_of_syllables;

            end

            fprintf('\n');
        end
        clear syllable_data filestats;

    end

    % PSD
    psdn = Xn_tot / Xn_frames;

    % syllable activity
    dataset_stats.syllable_activity = sum(dataset_stats.file_length)/dataset_stats.length;
    dataset_stats.nb_of_syllables = nb_of_syllables;
    dataset_stats.recording_time = dataset_stats.length*frame_shift/fs;
    dataset_stats.syllable_time = Xn_frames*frame_shift/fs;

    % save to data set file
    dataset_dir=handles.audiodir;
    save(dataset_filename,'dataset_content','dataset_dir','-v6');
    save(dataset_filename,'dataset_stats','-append','-v6');
    save(dataset_filename,'psdn','fs','-append','-v6');

end

% show_syllables
function show_syllables(handles,syllable_ndx)

    
    % make syllable patch
    syllable_gt = handles.syllable_data{2,syllable_ndx};
    syllable_duration=size(syllable_gt,2);
    syllable_patch_window=max(handles.patch_window,ceil(syllable_duration/2)*2);
    syllable_patch_gt = zeros(size(syllable_gt,1), syllable_patch_window);
    syllable_patch_window_start=floor(syllable_patch_window/2)-floor(syllable_duration/2);
    syllable_patch_gt(:, syllable_patch_window_start+1:syllable_patch_window_start+syllable_duration) = syllable_gt;
    syllable_fft = handles.syllable_data{3,syllable_ndx};

    syllable_fft_median=median(syllable_fft(:));
    syllable_fft_median=2*syllable_fft_median;
    syllable_patch_fft = syllable_fft_median*ones(size(syllable_fft,1), syllable_patch_window);
    syllable_duration=size(syllable_fft,2);
    syllable_patch_fft(:, syllable_patch_window_start+1:syllable_patch_window_start+syllable_duration) = syllable_fft;
    %syllable_duration
    %syllable_patch_window_start;
    %syllable_fft_median
    %_le
    %fprintf("processing Images \n ");
    % fft figure
    %axes(handles.syllable_axes_fft);
    syllable_patch_fft_dB=10*log10(abs(syllable_patch_fft(1:2:end,:)+1e-5)); % in dB
    %syllable_patch_fft
    fft_range_db1=-60;
    fft_range_db1_min=-30;
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
    usv = imresize(usv,[256 256]);
    %handles
    filename = fullfile(handles.image_dir,img_filename );
    imwrite(usv,filename,'png');
  
end

% move_syllable_slider
function move_syllable_slider(handles)

    if isempty(handles.syllable_stats)
        errordlg('Please select and process an audio file.','No audio file selected/processed')
    else
        syllable_ndx = round(get(handles.syllable_slider,'Value')./(get(handles.syllable_slider,'Max')) * (size(handles.syllable_stats,2)-1) + 1);
        % make syllable patch
        show_syllables(handles,syllable_ndx);
    end

end

% compute_csv_stats
function compute_csv_stats(wav_dir,wav_items,handles)

    init_waitbar = 1;
    cnt = 1;
    for fileID = 1:length(wav_items)

        if init_waitbar==1
           h = waitbar(0,'Initializing exporting process...');
           init_waitbar=0;
        end
        waitbar(cnt/(2*length(wav_items)),h,sprintf('Exporting syllable statistics to csv format... (file %i of %i)',(cnt-1)/2+1,length(wav_items)));

        [~, syllable_stats, ~, fs]=compute_musv(wav_dir,wav_items(fileID),handles,true);

        handles.syllable_stats = syllable_stats;
        handles.sample_frequency = fs;
        handles.filename = wav_items{fileID};
        export_csv_files_syllables(handles);

        cnt = cnt + 2;
    end
    if exist('h','var')
        close(h);
    end

end

% export_csv_files_syllables
function export_csv_files_syllables(handles)

    csvdir=fullfile(handles.audiodir,'CSV');
    if ~exist(csvdir,'dir')
      mkdir(csvdir)
    end

    % csv header
    csv_header=sprintf('%s,%s (sec),%s (sec),%s (kHz),%s (kHz),%s (kHz),%s (kHz),%s (kHz),%s (kHz),%s (msec),%s (msec),%s (dB),%s (dB)\n', ...
        'Syllable number', ...
        'Syllable start time', ...
        'Syllable end time', ...
        'starting frequency', ...
        'final frequency', ...
        'minimum frequency', ...
        'maximum frequency', ...
        'mean frequency', ...
        'frequency bandwidth', ...
        'syllable duration', ...
        'inter-syllable interval', ...
        'total syllable energy', ...
        'peak syllable amplitude');

    csvfile=strrep(fullfile(csvdir, strrep(handles.filename,'wav','csv')),'WAV','csv');
    fid = fopen(csvfile,'wt');
    fwrite(fid, csv_header);

    for syllable_ndx = 1:size(handles.syllable_stats,2)

        if handles.syllable_stats{14,syllable_ndx} == -100
            inter_syllable_interval='_';
        else
            inter_syllable_interval=sprintf('%.2f',handles.syllable_stats{14,syllable_ndx});
        end

        % print syllable information
        syllable_info=sprintf('%i,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%.2f,%.2f\n', ...
            syllable_ndx, ...
            handles.syllable_stats{8,syllable_ndx}, ...
            handles.syllable_stats{9,syllable_ndx}, ...
            handles.syllable_stats{2,syllable_ndx}, ...
            handles.syllable_stats{3,syllable_ndx}, ...
            handles.syllable_stats{4,syllable_ndx}, ...
            handles.syllable_stats{5,syllable_ndx}, ...
            handles.syllable_stats{10,syllable_ndx}, ...
            handles.syllable_stats{6,syllable_ndx}, ...
            handles.syllable_stats{13,syllable_ndx}, ...
            inter_syllable_interval, ...
            handles.syllable_stats{11,syllable_ndx}, ...
            handles.syllable_stats{12,syllable_ndx});
            fwrite(fid, syllable_info);
    end

    fclose(fid);

end

% export_csv_dataset_content
function export_csv_dataset_content(dataset_items, handles)

    csvdir=fullfile(handles.datasetdir,'CSV');
    if ~exist(csvdir)
      mkdir(csvdir)
    end

    % csv header
    csv_header=sprintf('%s,%s,%s\n', ...
        'data set', ...
        'file path', ...
        'file name');

    csvfile=fullfile(csvdir, sprintf('datasets_file_content.csv'));
    fid = fopen(csvfile,'wt');
    fwrite(fid, csv_header);

    for dataset_ndx = 1:length(dataset_items)
        datasetname=dataset_items{dataset_ndx};
        load(fullfile(handles.datasetdir,sprintf('%s.mat',datasetname)),'dataset_content','dataset_dir');

        fprintf('\nContent of dataset "%s"\n',sprintf('%s.mat',dataset_items{dataset_ndx}));
        fprintf('directory:   %s\n',dataset_dir);
        for k=1:length(dataset_content)
            % command window
            if k==1
                fprintf('files:       %s\n',dataset_content{k});
            else
                fprintf('             %s\n',dataset_content{k});
            end
            % print syllable information
            dataset_info=sprintf('%s,%s,%s\n', ...
                datasetname, ...
                dataset_dir, ...
                dataset_content{k});
            fwrite(fid, dataset_info);
        end

    end

    fclose(fid);

end

% repertoire_learning
function [bases, activations, bic, logL, syllable_similarity, syllable_correlation, repertoire_similarity, err, NbChannels, NbPatternFrames, NbUnits, NbIter, ndx_V, datasetNameRefine,msg] = repertoire_learning(handles, datasetName, NbUnits, NbChannels, NbIter, remove_ndx_refine, W_init)

    % Inputs
    if ~exist('NbChannels', 'var')
       NbChannels=64;
    end
    if ~exist('NbUnits', 'var')
       NbUnits=100;
    end
    if ~exist('NbIter', 'var')
       NbIter=100;
    end
    if ~exist('remove_ndx_refine', 'var')
        remove_ndx_refine=[];
    end
    if ~exist('W_init', 'var')
        W_init=[];
    end

    NbPatternFrames=handles.patch_window;
    fs=250000;
    frame_shift=floor(handles.frame_shift_ms*fs);
    min_nb_of_syllables=handles.repertoire_learning_min_nb_syllables_fac*NbUnits;

    % Gammatone features data
    load(fullfile(handles.datasetdir,datasetName),'dataset_content','dataset_dir','fs');
    flist=dataset_content;

    if isempty(remove_ndx_refine)
        h = waitbar(0,'Creating repertoire...');
    else
        h = waitbar(0,'Refining repertoire...');
    end

    % Accumulate GT sonogram frames
    V=[];
    backtrace.file=[];
    backtrace.start=[];
    fprintf('\nComputing input matrix\n');
    for fname = flist
         [~, filename]= fileparts(fname{1});
         syllable_file=fullfile(dataset_dir, sprintf('%s.mat', filename));

         if exist(syllable_file,'file'),
            fprintf('Loading MUSV of %s\n', filename);
            load(syllable_file,'syllable_data','filestats');
            syllable_data(3,:)={[]};

            % make input matrix
            Vi = mk_input_repertoire_learning(syllable_data(2,:), NbChannels, NbPatternFrames, filestats.TotNbFrames);
            V=[V Vi];
            backtrace.file=[backtrace.file syllable_data(1,:)];
            backtrace.start=[backtrace.start syllable_data(5,:)];

         else
            continue;
         end
    end

    if size(V,2) < min_nb_of_syllables
       msg=sprintf('WARNING: total number of detected syllables is less than %i\nBuild a smaller repertoire.\n', min_nb_of_syllables);
       errordlg(msg,'Repertoire learning stopped');
       bases={}; activations=[]; err=inf;
       return
    end

    % prepare data
    Vnorm = sqrt(sum(V.^2,1));
    remove_ndx = Vnorm<eps(max(Vnorm));
    V(:,remove_ndx)=[];

    % refinement
    msg='';
    datasetNameRefine=datasetName;
    if ~isempty(remove_ndx_refine)

        for fname = flist
            [~, filename]= fileparts(fname{1});
            syllable_file=fullfile(dataset_dir, sprintf('%s.mat', filename));
            audiofile=syllable_file;
            first_occur1=find(not(cellfun('isempty', strfind(backtrace.file,sprintf('%s.wav',filename)))),1);
            try
            	ndx1=not(cellfun('isempty', strfind(backtrace.file(remove_ndx_refine),sprintf('%s.wav',filename))));
                first_occur2=find(not(cellfun('isempty', strfind(backtrace.file,sprintf('%s.WAV',filename)))),1);
                ndx2=not(cellfun('isempty', strfind(backtrace.file(remove_ndx_refine),sprintf('%s.WAV',filename))));
                if ~isempty(first_occur1)
                    ndx_remove=remove_ndx_refine(ndx1)-first_occur1+1;
                elseif ~isempty(first_occur2)
                    ndx_remove=remove_ndx_refine(ndx2)-first_occur2+1;
                else
                    ndx_remove=[];
                end
            catch
                 msg='Files were already refined from another repertoire.';
                 bases=[];activations=[];err=[];NbChannels=[];
                 NbPatternFrames=[];NbUnits=[];NbIter=[];ndx_V=[];datasetNameRefine=[];
                 return;
            end


            % update syllable stats
            load(syllable_file,'syllable_data','syllable_stats','filestats');
            syllable_use=ones(1,filestats.nb_of_syllables);
            syllable_use(ndx_remove)=0;
            configpar=filestats.configpar;

            % update file stats
            syllable_stats_orig = syllable_stats;
            [syllable_data, syllable_stats, filestats] = syllable_activity_file_stats(handles, audiofile, filestats.TotNbFrames, syllable_data, syllable_use);
            filestats.configpar=configpar;
            save(syllable_file,'syllable_stats_orig','syllable_data','syllable_stats','filestats','-append','-v6');

        end

        % refine data set
        datasetNameRefine=sprintf('%s+.mat',datasetName);
        save(fullfile(handles.datasetdir,datasetNameRefine),'dataset_content','dataset_dir','fs','-v6');
        syllable_activity_stats_refine(handles, datasetNameRefine);
        delete(fullfile(handles.datasetdir,sprintf('%s.mat',datasetName)));

        % input matrix
        V(:,remove_ndx_refine)=[];
    end

    % learn base patterns
    if isempty(remove_ndx_refine)
        fprintf('Learning repertoire unit\n');
        waitbar(0.5,h,'Clustering into repertoire units...');
    else
        waitbar(0.5,h,'Updating repertoire units...');
    end

    [W,~,J,~,bic,logL,syllable_similarity,syllable_correlation,repertoire_similarity]=kmeans_clustering(V', NbUnits, NbChannels, NbIter, W_init);
    W=W';

    % track input syllables
    ndx_V=cell(NbUnits,1);
    for k=1:NbUnits
       ndx_V{k}=find(J==k);
    end

    % sort bases
    [number_of_calls]=hist(J,NbUnits);
    [~, ndx_sort]=sort(number_of_calls,'descend');
    W=W(:,ndx_sort);
    syllable_similarity=syllable_similarity(:,ndx_sort);
    syllable_correlation=syllable_correlation(:,ndx_sort);
    ndx_V=ndx_V(ndx_sort);
    activations=zeros(size(J));
    for k=1:NbUnits
       activations(J==ndx_sort(k))=k;
    end

    % prepare outputs
    bases=cell(NbUnits,1);
    for kk=1:NbUnits
        bases{kk} = reshape(W(1:NbChannels*(NbPatternFrames+1),kk),NbChannels,(NbPatternFrames+1));
    end

    % reconstruction error
    whs=W(:,activations);
    err = sum(sum( (V-whs).^2))./size(V,2);

    if exist('h','var')
        close(h);
    end

end

% mk_input_repertoire_learning
function [gt_sonogram_out] = mk_input_repertoire_learning(gt_sonogram_in, K, N, TotNbFrames)

    % parameters
    min_usv_activity_percentage=0.5;
    nb_of_syllables=length(gt_sonogram_in);

    % syllable activity
    usv_activity = size([gt_sonogram_in{:}],2)/TotNbFrames*100;
    if usv_activity < min_usv_activity_percentage
      fprintf('WARNING: file (less than %.2f%% of USV activity)\n',min_usv_activity_percentage);
    end

    % stacking syllable frames
    cnt=0;
    gt_sonogram_out=zeros(K*(N+1), nb_of_syllables);
    for k=1:nb_of_syllables,
        syllable=gt_sonogram_in{k};
        syllable_duration = size(gt_sonogram_in{k},2);
        if syllable_duration > N
            syllable=syllable(:,1:N);
            syllable_duration=N;
        end
        syllable_patch = zeros(K, N+1);
        cnt=cnt+1;
        gtcumsum=cumsum(sum(syllable,2));
        [~,gtcentern]=min(abs(gtcumsum-(gtcumsum(end)-gtcumsum(1))/2).^2);
        ndx_warp = mod([1:K]-(K/2-gtcentern),K)+1;
        syllable_patch(:, floor(N/2)-floor(syllable_duration/2)+1:floor(N/2)-floor(syllable_duration/2)+syllable_duration) = syllable(ndx_warp,:);
        gt_sonogram_out(:,cnt)=reshape(syllable_patch,(N+1)*K,1);
    end

    % trim
    gt_sonogram_out(:, end-(nb_of_syllables-cnt)+1:end) = [];

end

% kmeans_clustering
function [x,esq,j,model,bic,logL,syllable_similarity,syllable_correlation,repertoire_similarity] = kmeans_clustering(d,k,K,maxIter,x0)
%KMEANS Vector quantisation using K-means algorithm [X,ESQ,J]=(D,K,X0)
%Inputs:
% D contains data vectors (one per row)
% K is number of centres required
% X0 are the initial centres (optional)
%
%Outputs:
% X is output row vectors (K rows)
% ESQ is mean square error
% J indicates which centre each data vector belongs to

    [n,p] = size(d);

    if nargin<3
      maxIter=100;
    end

    if nargin<4
      K=p;
    end
    N=p/K;

    if nargin<5 || isempty(x0)
       x = d(ceil(rand(1,k)*n),:);
       x = d(1:floor(size(d,1)/k):n,:);
       x = x(1:k,:);
    else
       x=full(x0);
    end
    y = x+1;

    z=zeros(n,k);
    ItNr=0;
    fprintf('K-Means Clustering (K=%i)\n',k);
    fprintf('Max number of iterations: %i \n',maxIter);
    fprintf('iteration     ');
    while any(x(:) ~= y(:))
       ItNr=ItNr+1;

       if(ItNr>maxIter), break; end

%        logL = 0;
       %z=eucl_dx(d',x')';
       z = pdist2( d, x, 'cosine');

       [m,j] = min(z,[],2);
       y = x;
       for i=1:k
          s = j==i;
          if any(s)
            % z_k = mean(pdist2( d(s,:), x(i,:), 'cosine'));
            x(i,:) = mean(d(s,:),1);
          else
             q=find(m~=0);
             if isempty(q) break; end
             r=q(ceil(rand*length(q)));
             x(i,:) = d(r,:);
             m(r)=0;
             y=x+1;
          end
%           logL_k = sum(log(max(1-pdist2( d(s,:), x(i,:), 'cosine'),1e-15)));
%           [~,~,logL_k] = cosine_similarity(d(s,:), x(i,:));
%           logL = logL + logL_k;
       end
%        bic = -2*logL + k*log(n);
%        fprintf('\n Iteration %3d | Log-likelihood: %.3f | BIC: %.3f ...',ItNr, logL, bic);

       fprintf('\b\b\b\b %3d' , ItNr);
    end

    syllable_correlation=zeros(2,k);
    syllable_similarity=zeros(2,k);
    repertoire_cossim=[];
    z = pdist2( d, x, 'cosine');
    [m,j] = min(z,[],2);
    logL = 0;
    for i=1:k
      s = j==i;
      [cossim,~,logL_k] = cosine_similarity(d(s,:), x(i,:));
      repertoire_cossim = [repertoire_cossim  cossim];
      measure_corr = corr(d(s,:)',x(i,:)');
      syllable_similarity(1,i) = mean(1-cossim);
      syllable_similarity(2,i) = std(1-cossim);
      syllable_correlation(1,i) = mean(measure_corr);
      syllable_correlation(2,i) = std(measure_corr);
      logL = logL + logL_k;
    end
    bic = -2*logL + k*log(n);
    repertoire_similarity = mean(repertoire_cossim);
    repertoire_similarity(end+1) = std(repertoire_cossim);
    fprintf('\n Log-likelihood: %.3f | BIC: %.3f ...',logL, bic);
    esq=mean(m,1);
    fprintf('\n [done]\n');

    if nargout > 3
      model.mu=zeros(size(x));
      model.var=zeros(size(x));
      for i=1:k
          di=d(j==1,:);
          model.mu(i,:)=x(i,:);
          model.var(i,:)=var(di);
      end
    end
end

% cosine similarity
function [cossim, cossim_logl, cossim_logl_sum] = cosine_similarity(X,y)
    cossim = [];
    for k = 1:size(X,1)
        cossim(end+1) = X(k,:)*y'/(norm(X(k,:),2)*norm(y,2));
    end
    cossim_logl = log(max((cossim+1)/2,1e-15));
    cossim_logl_sum = sum(cossim_logl);
end

% export_csv_model_stats_repertoires
function export_csv_model_stats_repertoires(handles, repertoireNames)

    csvdir=fullfile(handles.repertoiredir,'CSV');
    if ~exist(csvdir,'dir')
      mkdir(csvdir)
    end

    % csv header
    csv_header=sprintf('%s,%s,%s,%s\n', ...
        'repertoire name', ...
        'overall repertoire modeling score', ...
        'average log-likelihood', ...
        'Bayesian Information Criterion (BIC)');

    csvfile=fullfile(csvdir, 'repertoire_modeling_info.csv');
    fid = fopen(csvfile,'wt');
    fwrite(fid, csv_header);

    for repertoireID = 1:length(repertoireNames)
        repertoireName=repertoireNames{repertoireID};
        repertoire_filename=fullfile(handles.repertoiredir,repertoireName);

        % load
        load(repertoire_filename,'bic','logL','repertoire_similarity');

        repertoire_info=sprintf('%s,%.3f,%.1f,%.1f\n', ...
                strrep(repertoireName,'.mat',''), ...
                repertoire_similarity(1), ...
                logL, ...
                bic);
        fwrite(fid, repertoire_info);
    end
    fclose(fid);
end

% export_csv_repertoires
function export_csv_repertoires(handles,repertoireNames)

    csvdir=fullfile(handles.repertoiredir,'CSV');
    if ~exist(csvdir,'dir')
      mkdir(csvdir)
    end

    % csv header
    csv_header=sprintf('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
        'repertoire unit (RU) number', ...
        'number of syllables', ...
        'syllable-to-centroid distance (mean)', ...
        'syllable-to-centroid distance (std)', ...
        'syllable-to-centroid correlation (mean)', ...
        'syllable-to-centroid correlation (std)', ...
        'final freq - start freq (+/- kHz)', ...
        'mean freq - start freq (+/- kHz)', ...
        'mean freq - final freq (+/- kHz)', ...
        'average frequency bandwidth (kHz)', ...
        'RU duration (msec)');
    csv_header1=sprintf('%s,%s,%s,%s,%s,%s\n', ...
        'dataset', ...
        'repertoire file', ...
        'syllable number', ...
        'syllable start time (sec)', ...
        'syllable end time (sec)', ...
        'repertoire unit (RU) number');

    for repertoireID = 1:length(repertoireNames)
        repertoireName=repertoireNames{repertoireID};
        repertoire_filename=fullfile(handles.repertoiredir,repertoireName);

        fprintf('Exporting repertoire %s \n', strrep(repertoireName,'.mat','') );

        % load
        load(repertoire_filename,'activations','NbUnits','ndx_V','datasetName','bic','logL','syllable_similarity','syllable_correlation','repertoire_similarity');
        H=activations';

        [number_of_calls]=hist(H,NbUnits);
        %[percentage_of_calls,~]=sort(number_of_calls./sum(number_of_calls)*100,'descend');
        number_of_calls=sort(number_of_calls,'descend');

        csvfile=fullfile(csvdir, strrep(repertoireName,'.mat','.csv'));
        fid = fopen(csvfile,'wt');
        fwrite(fid, csv_header);

        csvfile1=fullfile(csvdir, strrep(repertoireName,'.mat','_syllable_sequence.csv'));
        fid1 = fopen(csvfile1,'wt');
        fwrite(fid1, csv_header1);

        % Gammatone features data
        datasetfile = fullfile(handles.datasetdir,datasetName);
        if ~exist(sprintf('%s.mat', datasetfile),'file')
            datasetfile = fullfile(handles.datasetdir,sprintf('%s+',datasetName));
        end
        load(datasetfile,'dataset_content','dataset_dir','fs');
        flist=dataset_content;

        % Accumulate GT sonogram frames
        freq_start=[];
        freq_final=[];
        freq_min=[];
        freq_max=[];
        freq_mean=[];
        freq_bw=[];
        syl_tot_en=[];
        syl_dur=[];
        syl_cnts=0;
        repertoire_unit_sequence=cell(length(flist),1);
        fnameID = 1;
        for fname = flist
             [~, filename]= fileparts(fname{1});
             syllable_file=fullfile(dataset_dir, sprintf('%s.mat', filename));

             if exist(syllable_file,'file'),
                fprintf('Loading MUSV of %s\n', filename);

                load(syllable_file,'syllable_stats');
                nb_syllables = size(syllable_stats,2);
                repertoire_unit_sequence{fnameID} = activations(syl_cnts+1:syl_cnts + nb_syllables);
                syl_cnts = syl_cnts + size(syllable_stats,2);
                if ~strcmp(repertoire_filename(end-4:end),'+.mat') && strcmp(datasetfile(end),'+'),
                    load(syllable_file,'syllable_stats_orig');
                    syllable_stats=syllable_stats_orig;
                end

                freq_start=[freq_start syllable_stats{2,:}];
                freq_final=[freq_final syllable_stats{3,:}];
                freq_min=[freq_min syllable_stats{4,:}];
                freq_max=[freq_max syllable_stats{5,:}];
                freq_bw=[freq_bw syllable_stats{6,:}];
                freq_mean=[freq_mean syllable_stats{10,:}];
                syl_tot_en=[syl_tot_en syllable_stats{11,:}];
                syl_dur=[syl_dur syllable_stats{13,:}];

             else
                continue;
             end

             for syllableID = 1:nb_syllables
                 syllable_seq_info=sprintf('%s,%s,%i,%.4f,%.4f,%i\n', ...
                     datasetName, ...
                     filename, ...
                     syllableID, ...
                     syllable_stats{8,syllableID}, ...
                     syllable_stats{9,syllableID}, ...
                     repertoire_unit_sequence{fnameID}(syllableID));
                fwrite(fid1, syllable_seq_info);
             end

             fnameID = fnameID + 1;
        end
        fclose(fid1);

        plus_sign={'','+'};
        for unitID = 1:length(number_of_calls)

            % final - start
            % if final>>start and mean frequency between (upward)
            % if final<<start and mean frequency between (upward)
            % if final~=start and mean frequency between and bandwidth small (flat)
            % ...
            freq_final_minus_start=freq_final(ndx_V{unitID})-freq_start(ndx_V{unitID});
            [tmp1,tmp2]=hist(freq_final_minus_start,5); [tmp3,tmp4]=max(tmp1); freq_final_minus_start_avg=tmp2(tmp4);

            freq_mean_minus_start=freq_mean(ndx_V{unitID})-freq_start(ndx_V{unitID});
    %         [tmp1,tmp2]=hist(freq_mean_minus_start,5); [tmp3,tmp4]=max(tmp1); freq_mean_minus_start_avg=tmp2(tmp4);
            freq_mean_minus_start_avg=mean(freq_mean_minus_start);
            freq_min_minus_start=freq_min(ndx_V{unitID})-freq_start(ndx_V{unitID});
    %         [tmp1,tmp2]=hist(freq_min_minus_start,5); [tmp3,tmp4]=max(tmp1); freq_min_minus_start_avg=tmp2(tmp4);
            freq_min_minus_start_avg=mean(freq_min_minus_start);
            freq_max_minus_start=freq_max(ndx_V{unitID})-freq_start(ndx_V{unitID});
    %         [tmp1,tmp2]=hist(freq_max_minus_start,5); [tmp3,tmp4]=max(tmp1); freq_max_minus_start_avg=tmp2(tmp4);
            freq_max_minus_start_avg=mean(freq_max_minus_start);

            freq_mean_minus_final=freq_mean(ndx_V{unitID})-freq_final(ndx_V{unitID});
    %         [tmp1,tmp2]=hist(freq_mean_minus_final,5); [tmp3,tmp4]=max(tmp1); freq_mean_minus_final_avg=tmp2(tmp4);
            freq_mean_minus_final_avg=mean(freq_mean_minus_final);
            freq_min_minus_final=freq_min(ndx_V{unitID})-freq_final(ndx_V{unitID});
    %         [tmp1,tmp2]=hist(freq_min_minus_final,5); [tmp3,tmp4]=max(tmp1); freq_min_minus_final_avg=tmp2(tmp4);
            freq_min_minus_final_avg=mean(freq_min_minus_final);
            freq_max_minus_final=freq_max(ndx_V{unitID})-freq_final(ndx_V{unitID});
    %         [tmp1,tmp2]=hist(freq_max_minus_final,5); [tmp3,tmp4]=max(tmp1); freq_max_minus_final_avg=tmp2(tmp4);
            freq_max_minus_final_avg=mean(freq_max_minus_final);

            [tmp1,tmp2]=hist(freq_start(ndx_V{unitID}),5); [tmp3,tmp4]=max(tmp1); freq_start_avg=tmp2(tmp4);
            [tmp1,tmp2]=hist(freq_final(ndx_V{unitID}),5); [tmp3,tmp4]=max(tmp1); freq_final_avg=tmp2(tmp4);
            [tmp1,tmp2]=hist(freq_min(ndx_V{unitID}),5); [tmp3,tmp4]=max(tmp1); freq_min_avg=tmp2(tmp4);
            [tmp1,tmp2]=hist(freq_max(ndx_V{unitID}),5); [tmp3,tmp4]=max(tmp1); freq_max_avg=tmp2(tmp4);
            [tmp1,tmp2]=hist(freq_bw(ndx_V{unitID}),5); [tmp3,tmp4]=max(tmp1); freq_bw_avg=tmp2(tmp4);
            [tmp1,tmp2]=hist(freq_mean(ndx_V{unitID}),5); [tmp3,tmp4]=max(tmp1); freq_mean_avg=tmp2(tmp4);
            [tmp1,tmp2]=hist(syl_tot_en(ndx_V{unitID}),5); [tmp3,tmp4]=max(tmp1); syl_en_avg=tmp2(tmp4);
            [tmp1,tmp2]=hist(syl_dur(ndx_V{unitID}),5); [tmp3,tmp4]=max(tmp1); syl_dur_avg=tmp2(tmp4);

            % print unit information
            dataset_info=sprintf('%i,%i,%.4f,%.4f,%.4f,%.4f,%+.2f,%+.2f,%+.2f,%.2f,%.2f\n', ...
                unitID, ...
                number_of_calls(unitID), ...
                syllable_similarity(1,unitID), ...
                syllable_similarity(2,unitID), ...
                syllable_correlation(1,unitID), ...
                syllable_correlation(2,unitID), ...
                freq_final_minus_start_avg, ...
                freq_mean_minus_start_avg, ...
                freq_mean_minus_final_avg, ...
                freq_bw_avg, ...
                syl_dur_avg);
            fwrite(fid, dataset_info);
        end
        fclose(fid);
    end
end

% show_repertoire_figures
function show_repertoire_figures(handles,repertoireName)

    guihandle=handles.output;
    repertoiredir=handles.repertoiredir;

    % figure
    set(guihandle, 'HandleVisibility', 'off');
    %close all;
    set(guihandle, 'HandleVisibility', 'on');
    screenSize=get(0,'ScreenSize');
    defaultFigPos=get(0,'DefaultFigurePosition');
    repertoire_filename=fullfile(repertoiredir,repertoireName);

    % load
    load(repertoire_filename,'bases','activations','NbUnits','NbChannels','NbPatternFrames','ndx_V');

    figure('Position',[defaultFigPos(1) 0.90*screenSize(4)-defaultFigPos(4) defaultFigPos(3)*(1+fix(NbUnits/40)) defaultFigPos(4)]);

    W=bases;
    H=activations';

    [number_of_calls]=hist(H,NbUnits);
    linebases=W;

    NbRows=5;
    NbCols=floor(NbUnits/NbRows);
    linebases_mat=zeros(NbChannels*NbRows,(NbPatternFrames+1)*NbCols);
    for kk=1:NbRows
    for ll=1:NbCols
      base_unit_normalized = linebases{(NbRows-kk)*NbCols+ll}./max(max(linebases{(NbRows-kk)*NbCols+ll}));
      linebases_mat((kk-1)*NbChannels+1:kk*NbChannels,(ll-1)*(NbPatternFrames+1)+1:ll*(NbPatternFrames+1))=base_unit_normalized;
    end
    end
    imagesc(linebases_mat,[0 0.85]); axis xy; hold on;
    for kk=1:NbCols-1
    plot([(NbPatternFrames+1)*kk+1,(NbPatternFrames+1)*kk+1],[1 NbRows*NbChannels],'Color',[0.2 0.2 0.2],'LineWidth',1);
    end
    for kk=1:NbRows-1
    plot([1 size(linebases_mat,2)],[kk*NbChannels+1 kk*NbChannels+1 ],'Color',[0.2 0.2 0.2],'LineWidth',1);
    end
    cnt=0;
    for kk=NbRows-1:-1:0
      for jj=0:NbCols-1
          cnt=cnt+1;
          text(floor(NbPatternFrames*.05)+jj*(NbPatternFrames+1),(kk+1)*NbChannels-10,num2str(cnt),'Color','k','FontSize',handles.FontSize2,'fontweight','bold');
          text(jj*(NbPatternFrames+1)+floor(NbPatternFrames*.05),(kk+1)*NbChannels-(NbChannels-10),sprintf('%i',number_of_calls(cnt)),'Color','b','FontSize',handles.FontSize2,'fontweight','normal');
      end
    end
    set(gcf, 'Color', 'w');
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    %ylabel('Gammatone channels','FontSize',handles.FontSize1);
    colormap pink; colormap(flipud(colormap));

    [~,repertoire_titlename]=fileparts(repertoireName);
    repertoire_titlename=strtrim(regexprep(repertoire_titlename,'[_([{}()=''.(),;%{%}!@])]',' '));
    repertoire_titlename=strtrim(regexprep(repertoire_titlename,':','/'));
    title(sprintf('Syllable repertoire: %s',repertoire_titlename),'FontSize',handles.FontSize1,'fontweight','bold');
    set(gca, 'looseinset', get(gca, 'tightinset'));
    hold off;

    %xlabel('Frames','FontSize',handles.FontSize1);
    set(gca, 'FontSize',handles.FontSize2);

end

% refine_selected_repertoire
function refine_selected_repertoire(handles,repertoireName)

    repertoiredir=handles.repertoiredir;
    repertoire_filename=fullfile(repertoiredir,repertoireName);
    refined_repertoire_filename=strrep(repertoire_filename,'.mat','+.mat');

    NbIterRefine=10;

    choice='Yes';
    if exist(refined_repertoire_filename,'file') || ~isempty(strfind(repertoire_filename,'+.mat'));
    %     qstring='Repertoire was already refined. This operation will overwrite the existing repertoire. Would you like to continue refining?';
    %     choice=questdlg(qstring, 'Further repertoire refinement','Yes','No','No');
        qstring='Repertoire was already refined.';
        errordlg(qstring, 'Repertoire refined stopped');
        choice='No';
    end

    load(repertoire_filename,'datasetName');
    datasetNameRefine=fullfile(handles.datasetdir, sprintf('%s+.mat',datasetName));
    if exist(datasetNameRefine,'file')
        qstring='Dataset was already refined through a different repertoire.';
        qstring=sprintf('Dataset %s already exists. Probably refined through a different repertoire.', sprintf('%s+',datasetName)) ;
        errordlg(qstring, 'Multiple dataset refinements conflict');
        choice='No';
    end

    switch choice
        case 'No'
            return;
        case 'Yes'
            % load
            load(repertoire_filename,'bases','activations','NbUnits','NbChannels','NbPatternFrames','NbIter','dataset_dir','ndx_V','datasetName');

            x = inputdlg('Enter space-separated numbers:', sprintf('Units to be removed from repertoire %s:',strrep(repertoireName,'.mat','')),[1 80]);

            if isempty(x)
                return
            elseif isempty(str2num(x{:}))
                 errordlg('Invalid input.','Repertoire refinement stopped',[1 80]);
            else
                units = fix(str2num(x{:}));
                units = unique(units);

                if ~isempty(find(units > NbUnits | units <= 0, 1))
                    errordlg('Invalid input.','Repertoire refinement stopped',[1 80]);
                else

                    % remove elements from repertoire
                    remove_ndx_refine=vertcat(ndx_V{units});

                    % compute initial centroid for refined repertoire learning
                    bases_tmp=bases;
                    bases_init=bases;
                    bases_tmp(units)=[];
                    bases_init(units)=bases_tmp(1:length(units));
    %                 bases_init(units)=bases_tmp(end-length(units)+1:end);
                    W_init=zeros(NbChannels*(NbPatternFrames+1),NbUnits);
                    for k=1:NbUnits
                        W_init(:,k)=reshape(bases_init{k},NbChannels*(NbPatternFrames+1),1);
                    end
                    W_init=W_init';

                    % recompute repertoire
                    [bases, activations, bic, logL, syllable_similarity, syllable_correlation, repertoire_similarity, err, NbChannels, NbPatternFrames, NbUnits, NbIter, ndx_V, datasetNameRefined,msg] = repertoire_learning(handles, datasetName, NbUnits, NbChannels, NbIterRefine, remove_ndx_refine, W_init);
                    if isempty(bases)
                        if isempty(msg)
                            errordlg('Repertoire learning stopped. Too few syllables were detected in the audio data.','repertoire error');
                        else
                            errordlg(msg,'repertoire error');
                        end
                    else
                        % save
                        [~,datasetName]=fileparts(datasetNameRefined);
                        save(refined_repertoire_filename,'bases','activations','bic','logL','syllable_similarity','syllable_correlation','repertoire_similarity','NbUnits','NbChannels','NbPatternFrames','NbIter','dataset_dir','remove_ndx_refine','ndx_V','datasetName','-v6');
                        set(handles.repertoire_list,'value',1);
                        delete(repertoire_filename);
                    end
                end
            end
    end
end

% show_syllable_category_counts
function show_syllable_category_counts(handles,repertoireNames,NbCategories)

    guihandle=handles.output;

    csvdir=fullfile(handles.repertoiredir,'CSV');
    if ~exist(csvdir)
      mkdir(csvdir)
    end

    % csv header
    csv_header=sprintf('%s,%s,%s,%s,%s\n', ...
        'repertoire unit (RU) cluster', ...
        'number of RUs', ...
        'RU-to-centroid correlation (mean)', ...
        'RU-to-centroid correlation (std)', ...
        'total number of syllables');

    %figure
    set(guihandle, 'HandleVisibility', 'off');
    close all;
    set(guihandle, 'HandleVisibility', 'on');
    screenSize=get(0,'ScreenSize');
    defaultFigPos=get(0,'DefaultFigurePosition');
    basesunits=[];
    basesactivations=cell(length(repertoireNames),1);
    basesmap=cell(length(repertoireNames),1);

    for repertoireID = 1:length(repertoireNames)

        repertoire_filename = fullfile(handles.repertoiredir,repertoireNames{repertoireID});

        if ~exist(repertoire_filename)
            fprintf('Repertoire of data set does not exist. Build the repertoire.\n');
        else
            load(repertoire_filename,'bases','activations','NbUnits','NbChannels','NbPatternFrames');
        end
        basesmap{repertoireID}=[length(basesunits)+1:length(basesunits)+length(bases)];
        basesunits=[basesunits; bases];
        basesactivations{repertoireID}=activations;

    end

    %
    figure('Position',[defaultFigPos(1) 0.90*screenSize(4)-defaultFigPos(4) defaultFigPos(3)*(1+fix(NbCategories/40)) defaultFigPos(4)]);
    V=zeros(NbChannels*(NbPatternFrames+1), length(basesunits));
    for k=1:length(basesunits)
      V(:,k)=reshape(basesunits{k}, NbChannels*(NbPatternFrames+1), 1);
    end

    % meta clustering
    [W,J]=kmedoids(V, NbCategories);

    syllable_category_correlation=zeros(length(repertoireNames),2,NbCategories);
    nb_of_repertoire_units=zeros(NbCategories,length(repertoireNames));
    for repertoireID = 1:length(repertoireNames)

        d_repertoire=V(:,basesmap{repertoireID});
        J_repertoire=J(basesmap{repertoireID});
        for categoryID=1:NbCategories
            s = J_repertoire==categoryID;
            nb_of_repertoire_units(categoryID,repertoireID) = sum(s);
            if nb_of_repertoire_units(categoryID,repertoireID) == 0
                syllable_category_correlation(repertoireID,1,categoryID) = 0;
                syllable_category_correlation(repertoireID,2,categoryID) = 0;
            else
                measure_corr = corr(d_repertoire(:,s),W(:,categoryID));
                syllable_category_correlation(repertoireID,1,categoryID) = mean(measure_corr);
                syllable_category_correlation(repertoireID,2,categoryID) = std(measure_corr);
            end
        end
    end

    % prepare outputs
    linebases=cell(NbCategories,1);
    for kk=1:NbCategories
        linebases{kk} = reshape(W(1:NbChannels*(NbPatternFrames+1),kk),NbChannels,(NbPatternFrames+1));
    end

    NbRows=5;
    NbCols=floor(NbCategories/NbRows);
    linebases_mat=zeros(NbChannels*NbRows,(NbPatternFrames+1)*NbCols);
    for kk=1:NbRows
      for ll=1:NbCols
        base_unit_normalized = linebases{(NbRows-kk)*NbCols+ll}./max(max(linebases{(NbRows-kk)*NbCols+ll}));
        linebases_mat((kk-1)*NbChannels+1:kk*NbChannels,(ll-1)*(NbPatternFrames+1)+1:ll*(NbPatternFrames+1))=base_unit_normalized;
      end
    end
    imagesc(linebases_mat,[0 0.85]); axis xy; hold on;
    for kk=1:NbCols-1
      plot([(NbPatternFrames+1)*kk+1,(NbPatternFrames+1)*kk+1],[1 NbRows*NbChannels],'Color',[0.2 0.2 0.2],'LineWidth',1);
    end
    for kk=1:NbRows-1
      plot([1 size(linebases_mat,2)],[kk*NbChannels+1 kk*NbChannels+1 ],'Color',[0.2 0.2 0.2],'LineWidth',1);
    end
    cnt=0;
    for kk=NbRows-1:-1:0
        for jj=0:NbCols-1
            cnt=cnt+1;
            text(floor(0.05*NbPatternFrames)+jj*(NbPatternFrames+1),(kk+1)*NbChannels-10,num2str(cnt),'Color','k','FontSize',handles.FontSize2,'fontweight','normal');
        end
    end
    set(gcf, 'Color', 'w');
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    % ylabel('Gammatone channels','FontSize',handles.FontSize1);
    colormap pink; colormap(flipud(colormap));
    title(sprintf('Meta USV syllable repertoire'),'FontSize',handles.FontSize1,'fontweight','bold');
    hold off;
    set(gca, 'looseinset', get(gca, 'tightinset'));

    % figure
    xtickunits=2;
    colormap_mupet=load_colormap;
    activation_pie=zeros(NbCategories,length(repertoireNames));
    nb_of_calls_pie=zeros(NbCategories,length(repertoireNames));
    for repertoireID = 1:length(repertoireNames)

        lineactivity=basesactivations{repertoireID};
        Jline=J((repertoireID-1)*NbUnits+1:repertoireID*NbUnits);
        lineactivity_mapped=Jline(lineactivity);
        for k=1:NbCategories
          activation_pie(k,repertoireID)=sum(lineactivity_mapped==k);
        end
        nb_of_calls_pie(:,repertoireID)=activation_pie(:,repertoireID);
        activation_pie(:,repertoireID)=100*activation_pie(:,repertoireID)./sum(activation_pie(:,repertoireID));
    end

    for repertoireID = 1:length(repertoireNames)

        figure('Position',[defaultFigPos(1)+(repertoireID)*25 0.90*screenSize(4)-defaultFigPos(4)-(repertoireID)*25 defaultFigPos(3) defaultFigPos(4)]);

        % pie diagram
        h=bar(nb_of_calls_pie(:,repertoireID)',1,'grouped');
        bar_child=get(h,'Children');
        colormap(colormap_mupet);
        %set(bar_child,'CData',max(log(activation_pie(:,repertoireID)'),0));
        set(bar_child,'CData',max(log(min(activation_pie(:,repertoireID),25)'),0));
        set(gca,'XTick',[0:xtickunits:NbCategories]);
        xlim([.5 NbCategories+.5]);
        %ylim([0 40]);
        set(gca,'XGrid','off','YGrid','on','XMinorGrid','off');
        set(gca, 'Box', 'off', 'TickDir','out')
        repertoireName=strtrim(regexprep(strrep(repertoireNames{repertoireID},'.mat',''),'[_([{}()=''.(),;:%{%}!@])]',' '));
        title(sprintf('%s',strrep(repertoireName,'_',' ')),'FontSize',handles.FontSize1,'fontweight','bold');
        set(gca, 'FontSize',handles.FontSize2);
        set(gca, 'looseinset', get(gca, 'tightinset'));
    %     ylabel('Activity counts (%%)','FontSize',handles.FontSize1);
        ylabel('Number of syllables','FontSize',handles.FontSize1);
        xlabel('Repertoire units (RUs)','FontSize',handles.FontSize1);
        axis('square')

        % CSV
        csvfile=fullfile(csvdir, sprintf('%s_RU_cluster_counts_of_%i.csv',strrep(repertoireNames{repertoireID},'.mat',''),NbCategories));
        fid = fopen(csvfile,'wt');
        fwrite(fid, csv_header);
        for categoryID = 1:NbCategories
            % print category information
            syl_cat_cor_mn = syllable_category_correlation(repertoireID,1,categoryID);
            syl_cat_cor_st = syllable_category_correlation(repertoireID,2,categoryID);
            if syl_cat_cor_mn == 0 && syl_cat_cor_st == 0
                syl_cat_cor_mn_str = '-';
                syl_cat_cor_st_str = '-';
            elseif syl_cat_cor_st == 0;
                if abs(syl_cat_cor_mn - 1 ) < 1e-10 % not exactly 1
                    syl_cat_cor_mn_str = sprintf('%.0f',syl_cat_cor_mn);
                else
                    syl_cat_cor_mn_str = sprintf('%.4f',syl_cat_cor_mn);
                end
                syl_cat_cor_st_str = sprintf('%i',syl_cat_cor_st);
            else
                syl_cat_cor_mn_str = sprintf('%.4f',syl_cat_cor_mn);
                syl_cat_cor_st_str = sprintf('%.4f',syl_cat_cor_st);
            end

            category_info=sprintf('%i,%i,%s,%s,%i\n', ...
                categoryID, ...
                nb_of_repertoire_units(categoryID,repertoireID), ...
                syl_cat_cor_mn_str, ...
                syl_cat_cor_st_str, ...
                nb_of_calls_pie(categoryID,repertoireID));
            fwrite(fid, category_info);
        end
        fclose(fid);
    end
end

% kmedoids
function [medoids, label, energy, index] = kmedoids(X,k)
% X: d x n data matrix
% k: number of cluster

    v = dot(X,X,1);
    D1 = bsxfun(@plus,v,v')-2*(X'*X);

    dotproduct = bsxfun(@(x, y) x'*y,X,X);
    normX = sqrt(sum(X.^2,1));
    normproduct = normX'*normX;
    D2 = 1-dotproduct./normproduct;

    D = D1;

    n = size(X,2);
    [~,tmp]=sort(mean(D,2),'descend');
    [~, label] = min(D(tmp(1:k),:),[],1);
    last = 0;
    while any(label ~= last)
        [~, index] = min(D*sparse(1:n,label,1,n,k,n),[],1);
        last = label;
        [val, label] = min(D(index,:),[],1);
    end
    energy = sum(val);
    medoids = X(:,index);
end

% syllable_activity_stats_refine
function [syllables, fs, syllable_dur, syllable_distance, syllable_activity, syllable_count_per_minute, syllable_count_per_second] = syllable_activity_stats_refine(handles, datasetName, Nfft)

    if ~exist('Nfft', 'var')
        Nfft=512;
    end
    fs=250000;
    frame_shift=floor(handles.frame_shift_ms*fs);

    % Accumulate GT sonogram frames
    dataset_stats.syllable_dur=[];
    dataset_stats.syllable_distance=[];
    dataset_stats.syllable_activity=[];
    dataset_stats.syllable_count_per_minute=[];
    dataset_stats.syllable_count_per_second=[];
    dataset_stats.file_length=[];
    dataset_stats.filenames=[];
    dataset_stats.length=0;
    Xn_frames=0; Xn_tot=0;
    nb_of_syllables=0;

    % Gammatone features data
    load(fullfile(handles.datasetdir,datasetName),'dataset_content','dataset_dir');
    flist=dataset_content;

    for fname = flist
        [~, filename]= fileparts(fname{1});
        syllable_file=fullfile(dataset_dir, sprintf('%s.mat', filename));
        if exist(syllable_file,'file'),
            load(syllable_file,'syllable_data','syllable_stats','filestats');
            syllable_data(2,:)={[]};
            syllable_use=cell2mat(syllable_stats(1,:));
            syllable_data=syllable_data(:,syllable_use==1);

            if filestats.nb_of_syllables >=1

                % accumulate syllable stats
                dataset_stats.syllable_dur = [dataset_stats.syllable_dur filestats.syllable_dur];
                dataset_stats.syllable_distance = [dataset_stats.syllable_distance filestats.syllable_distance];
                dataset_stats.file_length = [dataset_stats.file_length; filestats.syllable_activity*filestats.TotNbFrames];
                dataset_stats.syllable_count_per_minute = [dataset_stats.syllable_count_per_minute; filestats.syllable_count_per_minute ];
                dataset_stats.syllable_count_per_second = [dataset_stats.syllable_count_per_second; filestats.syllable_count_per_second ];
                dataset_stats.length = dataset_stats.length + filestats.TotNbFrames;
                dataset_stats.filenames = [dataset_stats.filenames syllable_data(1,:)];

                % accumulate psd
                for syllableID = 1:filestats.nb_of_syllables
                    E=cell2mat(syllable_data(4,syllableID));
                    E(E==0)=1;
                    sumXn=sum(cell2mat(syllable_data(3,syllableID))./(ones(Nfft,1)*E),2);
                    Xn_tot = Xn_tot + sumXn;
                    Xn_frames=Xn_frames+length(E);
                end
                nb_of_syllables=nb_of_syllables+filestats.nb_of_syllables;

            end
        end
        clear syllable_data filestats;
    end

    % PSD
    psdn = Xn_tot / Xn_frames;

    % syllable activity
    dataset_stats.syllable_activity = sum(dataset_stats.file_length)/dataset_stats.length;
    dataset_stats.nb_of_syllables = nb_of_syllables;
    dataset_stats.recording_time = dataset_stats.length*frame_shift/fs;
    dataset_stats.syllable_time = Xn_frames*frame_shift/fs;

    % save to data set file
    save(fullfile(handles.datasetdir,datasetName),'dataset_stats','psdn','-append','-v6');

end

% compare_A_against_all_repertoires_match
function compare_repertoires_match(handles,repertoireNames,repertoireA)

    guihandle=handles.output;
    repertoiredir=handles.repertoiredir;

    % figure
    set(guihandle, 'HandleVisibility', 'off');
%     close all;
    set(guihandle, 'HandleVisibility', 'on');
    screenSize=get(0,'ScreenSize');
    defaultFigPos=get(0,'DefaultFigurePosition');

    cnt=0;

    for repertoireID = 1:length(repertoireNames)

        if strcmp(repertoireA,repertoireNames{repertoireID})
           continue;
        end
        cnt=cnt+1;

        repertoirePairNames={repertoireA,repertoireNames{repertoireID}};
        basesunits=cell(length(repertoirePairNames),1);
        basesactivations=cell(length(repertoirePairNames),1);
        basesnames=cell(length(repertoirePairNames),1);
        for repertoirePairID = 1:length(repertoirePairNames)

            repertoire_filename = fullfile(repertoiredir,repertoirePairNames{repertoirePairID});
            [~, repertoirename]=fileparts(repertoire_filename);

            if ~exist(repertoire_filename)
                fprintf('Repertoire of data set does not exist. Build the repertoire.\n');
            else
                load(repertoire_filename,'bases','activations','NbUnits','NbChannels','NbPatternFrames');
            end
            basesunits{repertoirePairID}=bases;
            basesactivations{repertoirePairID}=activations;
            basesnames{repertoirePairID}=strrep(repertoirename,'_',' ');

        end

        [mssim, mssim_diag , similarity_score_mean, sim_scores_highact, diag_score, ndx_permutation, ndx_permutation_second ] = ...
            repertoire_comparison(basesunits{1}, basesactivations{1}, basesunits{2}, basesactivations{2} );

        % figure
%         %figure('Position',[defaultFigPos(1)+(cnt-1) 0.90*screenSize(4)-defaultFigPos(4)-(cnt-1) defaultFigPos(3)*3 defaultFigPos(4)*1]);

%         subplot(1,3,2:3);

        %%% PRIMARY %%%
        figure;
        linebases=basesunits{1};
        [number_of_calls]=hist(basesactivations{1}',NbUnits);
        % permutate
        linebases=linebases(ndx_permutation);
        number_of_calls=number_of_calls(ndx_permutation);

        NbRows=5;
        NbCols=floor(NbUnits/NbRows);
        linebases_mat=zeros(NbChannels*NbRows,(NbPatternFrames+1)*NbCols);
        for kk=1:NbRows
        for ll=1:NbCols
          base_unit_normalized = linebases{(NbRows-kk)*NbCols+ll}./max(max(linebases{(NbRows-kk)*NbCols+ll}));
          linebases_mat((kk-1)*NbChannels+1:kk*NbChannels,(ll-1)*(NbPatternFrames+1)+1:ll*(NbPatternFrames+1))=base_unit_normalized;
        end
        end
        hsubplot(2)=imagesc(linebases_mat,[0 0.85]); axis xy; hold on;
        for kk=1:NbCols-1
        plot([(NbPatternFrames+1)*kk+1,(NbPatternFrames+1)*kk+1],[1 NbRows*NbChannels],'Color',[0.2 0.2 0.2],'LineWidth',1);
        end
        for kk=1:NbRows-1
        plot([1 size(linebases_mat,2)],[kk*NbChannels+1 kk*NbChannels+1 ],'Color',[0.2 0.2 0.2],'LineWidth',1);
        end
        cnt=0;
        for kk=NbRows-1:-1:0
          for jj=0:NbCols-1
              cnt=cnt+1;
              text(floor(NbPatternFrames*.05)+jj*(NbPatternFrames+1),(kk+1)*NbChannels-10,sprintf('%i (%i)', cnt, ndx_permutation(cnt)),'Color','k','FontSize',handles.FontSize2,'fontweight','bold');
              text(jj*(NbPatternFrames+1)+floor(NbPatternFrames*.05),(kk+1)*NbChannels-(NbChannels-10),sprintf('%i',number_of_calls(cnt)),'Color','b','FontSize',handles.FontSize2,'fontweight','normal');

          end
        end
        set(gcf, 'Color', 'w');
        set(gca,'XTick',[]);
        set(gca,'YTick',[]);
%         ylabel('Gammatone channels','FontSize',handles.FontSize1);
        colormap pink; colormap(flipud(colormap));

        title(sprintf('Syllable repertoire: %s',strtrim(regexprep(basesnames{1},'[_([{}()=''.(),;:%{%}!@])]',' '))),'FontSize',handles.FontSize1,'fontweight','bold');
        set(gca, 'looseinset', get(gca, 'tightinset'));
        hold off;

%         xlabel('Frames','FontSize',handles.FontSize1);
        set(gca, 'FontSize',handles.FontSize2);
        freezeColors;

        %%% SECOND %%%
        figure;
        linebases=basesunits{2};
        [number_of_calls]=hist(basesactivations{2}',NbUnits);
        % permutate
        linebases=linebases(ndx_permutation_second);
        number_of_calls=number_of_calls(ndx_permutation_second);

        NbRows=5;
        NbCols=floor(NbUnits/NbRows);
        linebases_mat=zeros(NbChannels*NbRows,(NbPatternFrames+1)*NbCols);
        for kk=1:NbRows
        for ll=1:NbCols
          base_unit_normalized = linebases{(NbRows-kk)*NbCols+ll}./max(max(linebases{(NbRows-kk)*NbCols+ll}));
          linebases_mat((kk-1)*NbChannels+1:kk*NbChannels,(ll-1)*(NbPatternFrames+1)+1:ll*(NbPatternFrames+1))=base_unit_normalized;
        end
        end
        hsubplot(2)=imagesc(linebases_mat,[0 0.85]); axis xy; hold on;
        for kk=1:NbCols-1
        plot([(NbPatternFrames+1)*kk+1,(NbPatternFrames+1)*kk+1],[1 NbRows*NbChannels],'Color',[0.2 0.2 0.2],'LineWidth',1);
        end
        for kk=1:NbRows-1
        plot([1 size(linebases_mat,2)],[kk*NbChannels+1 kk*NbChannels+1 ],'Color',[0.2 0.2 0.2],'LineWidth',1);
        end
        cnt=0;
        for kk=NbRows-1:-1:0
          for jj=0:NbCols-1
              cnt=cnt+1;
              text(floor(NbPatternFrames*.05)+jj*(NbPatternFrames+1),(kk+1)*NbChannels-10,sprintf('%i (%i)', cnt, ndx_permutation_second(cnt)),'Color','k','FontSize',handles.FontSize2,'fontweight','bold');
              text(jj*(NbPatternFrames+1)+floor(NbPatternFrames*.05),(kk+1)*NbChannels-(NbChannels-10),sprintf('%i',number_of_calls(cnt)),'Color','b','FontSize',handles.FontSize2,'fontweight','normal');

          end
        end
        set(gcf, 'Color', 'w');
        set(gca,'XTick',[]);
        set(gca,'YTick',[]);
%         ylabel('Gammatone channels','FontSize',handles.FontSize1);
        colormap pink; colormap(flipud(colormap));

        title(sprintf('Syllable repertoire: %s',strtrim(regexprep(basesnames{2},'[_([{}()=''.(),;:%{%}!@])]',' '))),'FontSize',handles.FontSize1,'fontweight','bold');
        set(gca, 'looseinset', get(gca, 'tightinset'));
        hold off;

%         xlabel('Frames','FontSize',handles.FontSize1);
        set(gca, 'FontSize',handles.FontSize2);
        freezeColors;

%         subplot(1,3,1);
        figure;
        colormap_mupet=load_colormap;
        colormap(colormap_mupet);
        cmap_sim=colormap_mupet;
        barcolors=colormap_mupet([1:floor(size(colormap_mupet,1)/size(sim_scores_highact,1)):size(colormap_mupet,1)],:);
        ipfac=1;
        mssim_diag_show=mssim_diag{6};

        ipfac=4;
        mssim_diag_show=padarray(mssim_diag{6},[1,1],0,'both');
        mssim_diag_show=interp2(mssim_diag_show,ipfac,'nearest'); %interpolate
        mssim_diag_show=mssim_diag_show(ipfac*2+1:end-ipfac*2-1,ipfac*2+1:end-ipfac*2-1);

        hsubplot(1)=imagesc(mssim_diag_show,[.0 1]);
        colormap(colormap_mupet);
        axis('square');
        set(gca,'XTick',[0:NbUnits/5*ipfac^2:NbUnits*ipfac^2]) % frequency
        set(gca,'XTickLabel',[0:NbUnits/5:NbUnits]) % frequency
        set(gca,'YTick',[0:NbUnits/5*ipfac^2:NbUnits*ipfac^2]) % frequency
        set(gca,'YTickLabel',[0:NbUnits/5:NbUnits]) % frequency
        ylabel(sprintf('Repertoire %s', basesnames{1}),'FontSize',handles.FontSize1,'FontWeight','bold');
        xlabel(sprintf('Repertoire %s', basesnames{2}),'FontSize',handles.FontSize1,'FontWeight','bold');
        set(gca, 'FontSize',handles.FontSize2);
        hcb=colorbar;
        set(get(hcb,'Title'),'String','Pearson''s correlation','FontSize',handles.FontSize2);
        axvals=axis;
        % text(axvals(1)+(axvals(2)-axvals(1))/2,axvals(3)+(axvals(4)-axvals(3))/2,{'MUPET version 1.0', '(unreleased)'},'Color',[0.5 0.5 0.5],'FontSize',handles.FontSize1+10,'HorizontalAlignment','center','Rotation',45);

        csvdir=fullfile(handles.repertoiredir,'CSV');
        if ~exist(csvdir)
          mkdir(csvdir)
        end

        % csv header
        csv_header=sprintf(',%s\n', ...
            'Pearson''s correlation (diagonal)');

        csvfile=fullfile(csvdir,sprintf('Repertoire_%s_vs_%s_comparison_with_best_match_sorting.csv',sprintf('%s', basesnames{1}),sprintf('%s', basesnames{2})));
        fid = fopen(csvfile,'wt');
        fwrite(fid, csv_header);

        mssim_diag_only = diag(mssim_diag{6});
        for diagID = 1:length(mssim_diag_only)
            fwrite(fid, sprintf('%i, %.3f\n',diagID, mssim_diag_only(diagID)));
        end
        fclose(fid);

    end
end

% compare_repertoires_activity
function compare_repertoires_activity(handles,repertoireNames,repertoireA)

    guihandle=handles.output;
    repertoiredir=handles.repertoiredir;

    csvdir=fullfile(handles.repertoiredir,'CSV');
    if ~exist(csvdir)
      mkdir(csvdir)
    end

    % csv header
    csv_header=sprintf('%s,%s,%s,%s,%s,%s\n', ...
        'dataset', ...
        'similarity score (top 5%)', ...
        'similarity score (Q1)', ...
        'similarity score (median)', ...
        'similarity score (Q3)', ...
        'similarity score (top 95%)');

    csvfile=fullfile(csvdir,sprintf('%s_vs_all_repertoire_comparison.csv',strrep(repertoireA,'.mat','')));
    fid = fopen(csvfile,'wt');
    fwrite(fid, csv_header);

    % csv header detailed
    csv_header_detailed=sprintf('%s','dataset');
    for percID=1:100
        csv_header_detailed=sprintf('%s',csv_header_detailed,sprintf('%s',sprintf(',%i%% PCTL',percID)));
    end
    csv_header_detailed=sprintf('%s\n',csv_header_detailed);
    csvfile_detailed=fullfile(csvdir,sprintf('%s_vs_all_repertoire_comparison_detailed.csv',strrep(repertoireA,'.mat','')));
    fid_detailed = fopen(csvfile_detailed,'wt');
    fwrite(fid_detailed, csv_header_detailed);

    % figure
    set(guihandle, 'HandleVisibility', 'off');
    %close all;
    set(guihandle, 'HandleVisibility', 'on');
    screenSize=get(0,'ScreenSize');
    defaultFigPos=get(0,'DefaultFigurePosition');
    basesunits=cell(length(repertoireNames),1);
    basesactivations=cell(length(repertoireNames),1);
    basesnames=cell(length(repertoireNames),1);
    for repertoireID = 1:length(repertoireNames)

        repertoire_filename = fullfile(repertoiredir,repertoireNames{repertoireID});
        [~, repertoirename]=fileparts(repertoire_filename);

        if ~exist(repertoire_filename)
            fprintf('Repertoire of data set does not exist. Build the repertoire.\n');
        else
            load(repertoire_filename,'bases','activations','NbUnits','NbChannels','NbPatternFrames');
        end
        basesunits{repertoireID}=bases;
        basesactivations{repertoireID}=activations;
        basesnames{repertoireID}=strrep(repertoirename,'_',' ');

    end

    ndx_A=find(strcmp(repertoireNames,repertoireA));
    ndx_comparing=find(~strcmp(repertoireNames,repertoireA));
    filename_A=basesnames(ndx_A);
    filenames_comparingrepertoire=basesnames(ndx_comparing);
    sim_scores=[];
    sim_scores_highact=[];
    sim_scores_highact_detailed=[];
    diag_scores=[];
    for repertoireID = 1:length(ndx_comparing)
        [mssim, mssim_diag , similarity_score_mean, similarity_score_highactivity, diag_score, ~, ~, similarity_score_highactivity_detailed ] = ...
            repertoire_comparison(basesunits{ndx_A}, basesactivations{ndx_A}, basesunits{ndx_comparing(repertoireID)}, basesactivations{ndx_comparing(repertoireID)} );
        sim_scores = [sim_scores; similarity_score_mean];
        sim_scores_highact = [sim_scores_highact; similarity_score_highactivity];
        sim_scores_highact_detailed = [sim_scores_highact_detailed; similarity_score_highactivity_detailed];
        diag_scores = [diag_scores; diag_score'];
    end

    % figure
    figure('Position',[defaultFigPos(1) 0.90*screenSize(4)-defaultFigPos(4) defaultFigPos(3)*(1+fix(length(ndx_comparing)/16)) defaultFigPos(4)]);
    minY=0.65;
    maxY=1.0;
    rec_width=.2;
    colormap_mupet=load_colormap;
    colormap(colormap_mupet);
    barcolors=colormap_mupet([1:floor(size(colormap_mupet,1)/size(sim_scores_highact,1)):size(colormap_mupet,1)],:);
    barcolors=0.5*ones(size(barcolors));
    bwqs=cell(1,size(sim_scores_highact,1));
    bwq=cell(1,size(sim_scores_highact,1));
    bwmin=cell(1,size(sim_scores_highact,1));
    bwmax=cell(1,size(sim_scores_highact,1));
    bwcnt=cell(1,size(sim_scores_highact,1));
    for barID=1:size(sim_scores_highact,1)
       bwqs{barID} = min(sim_scores_highact(barID, 4),sim_scores_highact(barID, 2));
       bwq{barID} = abs(sim_scores_highact(barID, 2) - sim_scores_highact(barID, 4));
       bwmin{barID} = sim_scores_highact(barID, 5);
       minY = min(minY,bwmin{barID}-0.025);
       bwmax{barID} = sim_scores_highact(barID, 1);
       bwcnt{barID} = sim_scores_highact(barID, 3);
       % print comparison information
       dataset_info=sprintf('%s,%.2f,%.2f,%.2f,%.2f,%.2f\n', ...
         strrep(filenames_comparingrepertoire{barID},' ','_'), ...
         sim_scores_highact(barID, 1), ...
         sim_scores_highact(barID, 2), ...
         sim_scores_highact(barID, 3), ...
         sim_scores_highact(barID, 4), ...
         sim_scores_highact(barID, 5));
       fwrite(fid, dataset_info);

       % print detailed comparison information
       dataset_info_detailed=sprintf('%s',strrep(filenames_comparingrepertoire{barID},' ','_'));
       fwrite(fid_detailed, dataset_info_detailed);
       for percID=1:100
           dataset_info_detailed=sprintf(',%.2f',sim_scores_highact_detailed(barID, percID));
           fwrite(fid_detailed, dataset_info_detailed);
       end
       fwrite(fid_detailed, sprintf('\n',''));

    end

    [~,ndx_sort]=sort(cell2mat(bwcnt),'descend');
    filenames_comparingrepertoire=filenames_comparingrepertoire(ndx_sort);
    for ndx=1:size(sim_scores_highact,1)
        barID = ndx_sort(ndx);
        rectangle('Position',[ndx-rec_width/2 bwqs{barID} rec_width bwq{barID}],'FaceColor',barcolors(barID,:)); hold on;
        plot([ndx], [bwmin{barID}], '+', 'MarkerEdgeColor','k', 'MarkerFaceColor','k');
        plot([ndx], [bwmax{barID}], '*', 'MarkerEdgeColor','k', 'MarkerFaceColor','w');
        plot([ndx-rec_width ndx+rec_width], [bwcnt{barID} bwcnt{barID}], 'k', 'LineWidth', 2);
    end
    grid on
    set(gca, 'Box', 'off', 'TickDir','out')
    ylim([minY maxY]);
    xlim([0.5 size(sim_scores_highact,1)+.5]);
%     axis('square');
    set(gca,'xtick',[1:length(filenames_comparingrepertoire)]);
    set(gca,'XTickLabel',filenames_comparingrepertoire);
    set(gca,'ytick',[0:0.1:maxY]);
    set(gca,'yTickLabel',[0:0.1:maxY]);
    ylabel('Similarity score (%)', 'FontSize',handles.FontSize1);
    title(filename_A,'FontSize',handles.FontSize1, 'FontWeight','bold');
    set(gca, 'FontSize',handles.FontSize2);
    hold off
    % text(axvals(1)+(axvals(2)-axvals(1))/2,axvals(3)+(axvals(4)-axvals(3))/2,{'MUPET version 1.0', '(unreleased)'},'Color',[0.9 0.9 0.9],'FontSize',handles.FontSize1+10,'HorizontalAlignment','center','Rotation',45);

end

% repertoire_comparison
function [mssim, mssim_diag, similarity_score_mean, similarity_score_highactivity, diag_ssim, ndx_permutation, ndx_permutation_second, similarity_score_highactivity_detailed] = repertoire_comparison(W1, H1, W2, H2 )

    % parameters
    rDim = length(W1);

    % rescale range
    L=255;
    W1max = max(cellfun(@(x) max(max(x)), W1, 'UniformOutput',1));
    W2max = max(cellfun(@(x) max(max(x)), W2, 'UniformOutput',1));

    % rerank based on occurency
    [activations_tot]=hist(H1,rDim);
    [~,ondx]=sort(activations_tot./sum(activations_tot)*100,'descend');
    activations_cumsum=(cumsum(activations_tot./sum(activations_tot)*100));
    [~, nbestMin]=min(abs(activations_cumsum-5));
    [~, nbest25]=min(abs(activations_cumsum-25));
    [~, nbest50]=min(abs(activations_cumsum-50));
    [~, nbest75]=min(abs(activations_cumsum-75));
    [~, nbestMax]=min(abs(activations_cumsum-95));

    nbestPerc=zeros(1,100);
    for percID = 1:100
        [~, nbestPerc(percID)]=min(abs(activations_cumsum-percID));
    end

    nbestAll=length(activations_cumsum);
    H1=H1(ondx);
    W1=W1(ondx);
    [activations_tot]=hist(H2,rDim);
    [~,ondx]=sort(activations_tot./sum(activations_tot)*100,'descend');
    H2=H2(ondx);
    W2=W2(ondx);

    % smooth
    %W1=cellfun(@(x) smooth2(x,1), W1,'uni',0);
    %W2=cellfun(@(x) smooth2(x,1), W2,'uni',0);

    % reshape
    W1=cell2mat( cellfun(@(x) reshape(x,prod(size(x)),1), W1,'uni',0)');
    W2=cell2mat( cellfun(@(x) reshape(x,prod(size(x)),1), W2,'uni',0)');

    % correlation distance
    measure_corr = bsxfun(@(x, y) corr(x,y), W1, W2);

    % cosine distance
    dotproduct = bsxfun(@(x, y) x'*y, W1, W2);
    normW1 = sqrt(sum(W1.^2,1));
    normW2 = sqrt(sum(W2.^2,1));
    normproduct = normW1'*normW2;
    measure_cosd = dotproduct./normproduct;

    mssim  = ( measure_cosd + measure_corr )./2 ;
    mssim  = ( measure_corr )./1 ;
    [similarity_score_mean, mssim_diag, diag_ssim, ndx_permutation, ndx_permutation_second ] = diag_score( mssim );
    similarity_score_highactivity=zeros(1,3);
    mssim_diag=cell(1,5);
    [similarity_score_highactivity(1), mssim_diag{1}] = highactivity_score(mssim, nbestMin);
    [similarity_score_highactivity(2), mssim_diag{2}] = highactivity_score(mssim, nbest25);
    [similarity_score_highactivity(3), mssim_diag{3}] = highactivity_score(mssim, nbest50);
    [similarity_score_highactivity(4), mssim_diag{4}]= highactivity_score(mssim, nbest75);
    [similarity_score_highactivity(5), mssim_diag{5}]= highactivity_score(mssim, nbestMax);
    [similarity_score_highactivity(6), mssim_diag{6}]= highactivity_score(mssim, nbestAll);

    similarity_score_highactivity_detailed=zeros(1,100);
    for percID=1:100
       similarity_score_highactivity_detailed(percID) = highactivity_score(mssim, nbestPerc(percID));
    end
end

% normalize
function Anorm = normalize(A)
    Anorm = (A - min(min(A)))/(max(max(A)) - min(min(A)));
end

% diag_score
function [sim_score_mean, mssim_diag, diag_ssim, ndx_permutation, ndx_permutation_second] = diag_score(mssim)

    % diagonalize
    rDim=size(mssim,1);
    tmp = mssim;
    mssim_diag = zeros(size(mssim));
    for l=1:rDim,
        [val1,ndx1]=max(tmp);
        [val2,ndx2]=max(val1);
        col = ndx2;
        row = ndx1(ndx2);
        mssim_diag(row+(l-1),l:end)=tmp(row, :);
        mssim_diag(l:end, col+(l-1))=tmp(:, col);
        mssim_diag(:,[l col+(l-1)])=mssim_diag(:,[col+(l-1) l]);
        mssim_diag([l row+(l-1)],:)=mssim_diag([row+(l-1) l],:);
        tmp(row, :)=[];
        tmp(:, col)=[];
    end;

    % main
    ndx_permutation=zeros(rDim,1);
    for l=1:rDim,
        ndx=find(sum(mssim==mssim_diag(l,l),2));
        k=0;
        while k<length(ndx)
            if ~ismember(ndx_permutation,ndx(k+1))
                ndx_permutation(l)=ndx(k+1);
            end
            k=k+1;
        end
    end

    % comparing
    ndx_permutation_second=zeros(rDim,1);
    for l=1:rDim,
        ndx=find(sum(mssim==mssim_diag(l,l),1));
        k=0;
        while k<length(ndx)
            if ~ismember(ndx_permutation_second,ndx(k+1))
                ndx_permutation_second(l)=ndx(k+1);
            end
            k=k+1;
        end
    end

    diag_energy = sum(diag(mssim_diag))./rDim;
    tot_energy = (sum(sum(mssim_diag))-diag_energy)./(rDim*rDim-rDim);
    diag_ssim = diag(mssim_diag);
    sim_score_mean=[];
    sim_score_mean(3) = mean(diag_ssim(1:floor(1.0*rDim)));
    sim_score_mean(2) = mean(diag_ssim(1:floor(0.75*rDim)));
    sim_score_mean(1) = mean(diag_ssim(1:floor(0.5*rDim)));

end

% highactivity_score
function [sim_score_highactivity, mssim_diag] = highactivity_score(mssim, nbest)

    % diagonalize
    tmp = mssim(1:nbest, 1:end);
    mssim_diag = zeros(nbest, size(tmp,2));
    mssim_diag_diag = [];
    for l=1:nbest,
        if l==nbest,
          ndx1 = ones(size(tmp,2),1);
          val1 = tmp;
        else
          [val1,ndx1]=max(tmp);
        end
        [val2,ndx2]=max(val1);
        col = ndx2;
        row = ndx1(ndx2);
        mssim_diag_diag = [ mssim_diag_diag tmp(row, col)];
        mssim_diag(row+(l-1),l:end)=tmp(row, :);
        mssim_diag(l:end, col+(l-1))=tmp(:, col);
        mssim_diag(:,[l col+(l-1)])=mssim_diag(:,[col+(l-1) l]);
        mssim_diag([l row+(l-1)],:)=mssim_diag([row+(l-1) l],:);
        tmp(row, :)=[];
        tmp(:, col)=[];
    end;


    mssim_diag=mssim_diag(1:nbest, 1:nbest);
    sim_score_highactivity = mean( mssim_diag_diag );

end

% freezeColors
%{
function freezeColors(varargin)

    appdatacode = 'JRI__freezeColorsData';

    [h, nancolor] = checkArgs(varargin);

    %gather all children with scaled or indexed CData
    cdatah = getCDataHandles(h);

    %current colormap
    cmap = colormap;
    nColors = size(cmap,1);
    cax = caxis;

    % convert object color indexes into colormap to true-color data using
    %  current colormap
    for hh = cdatah',

        g = get(hh);

        %preserve parent axis clim
        parentAx = getParentAxes(hh);
        originalClim = get(parentAx, 'clim');

        %   Note: Special handling of patches: For some reason, setting
        %   cdata on patches created by bar() yields an error,
        %   so instead we'll set facevertexcdata instead for patches.
        if ~strcmp(g.Type,'patch'),
            cdata = g.CData;
        else
            cdata = g.FaceVertexCData;
        end

        %get cdata mapping (most objects (except scattergroup) have it)
        if isfield(g,'CDataMapping'),
            scalemode = g.CDataMapping;
        else
            scalemode = 'scaled';
        end

        %save original indexed data for use with unfreezeColors
        siz = size(cdata);
        setappdata(hh, appdatacode, {cdata scalemode});

        %convert cdata to indexes into colormap
        if strcmp(scalemode,'scaled'),
            %4/19/06 JRI, Accommodate scaled display of integer cdata:
            %       in MATLAB, uint * double = uint, so must coerce cdata to double
            %       Thanks to O Yamashita for pointing this need out
            idx = ceil( (double(cdata) - cax(1)) / (cax(2)-cax(1)) * nColors);
        else %direct mapping
            idx = cdata;
            %10/8/09 in case direct data is non-int (e.g. image;freezeColors)
            % (Floor mimics how matlab converts data into colormap index.)
            % Thanks to D Armyr for the catch
            idx = floor(idx);
        end

        %clamp to [1, nColors]
        idx(idx<1) = 1;
        idx(idx>nColors) = nColors;

        %handle nans in idx
        nanmask = isnan(idx);
        idx(nanmask)=1; %temporarily replace w/ a valid colormap index

        %make true-color data--using current colormap
        realcolor = zeros(siz);
        for i = 1:3,
            c = cmap(idx,i);
            c = reshape(c,siz);
            c(nanmask) = nancolor(i); %restore Nan (or nancolor if specified)
            realcolor(:,:,i) = c;
        end

        %apply new true-color color data

        %true-color is not supported in painters renderer, so switch out of that
        if strcmp(get(gcf,'renderer'), 'painters'),
            set(gcf,'renderer','zbuffer');
        end

        %replace original CData with true-color data
        if ~strcmp(g.Type,'patch'),
            set(hh,'CData',realcolor);
        else
            set(hh,'faceVertexCData',permute(realcolor,[1 3 2]))
        end

        %restore clim (so colorbar will show correct limits)
        if ~isempty(parentAx),
            set(parentAx,'clim',originalClim)
        end
    end

end

%}
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

% checkArgs
function [h, nancolor] = checkArgs(args)
% checkArgs  Validate input arguments to freezeColors

    nargs = length(args);
    error(nargchk(0,3,nargs,'struct'))

    %grab handle from first argument if we have an odd number of arguments
    if mod(nargs,2),
        h = args{1};
        if ~ishandle(h),
            error('JRI:freezeColors:checkArgs:invalidHandle',...
                'The first argument must be a valid graphics handle (to an axis)')
        end
        % 4/2010 check if object to be frozen is a colorbar
        if strcmp(get(h,'Tag'),'Colorbar'),
          if ~exist('cbfreeze.m'),
            warning('JRI:freezeColors:checkArgs:cannotFreezeColorbar',...
                ['You seem to be attempting to freeze a colorbar. This no longer'...
                'works. Please read the help for freezeColors for the solution.'])
          else
            cbfreeze(h);
            return
          end
        end
        args{1} = [];
        nargs = nargs-1;
    else
        h = gca;
    end

    %set nancolor if that option was specified
    nancolor = [nan nan nan];
    if nargs == 2,
        if strcmpi(args{end-1},'nancolor'),
            nancolor = args{end};
            if ~all(size(nancolor)==[1 3]),
                error('JRI:freezeColors:checkArgs:badColorArgument',...
                    'nancolor must be [r g b] vector');
            end
            nancolor(nancolor>1) = 1; nancolor(nancolor<0) = 0;
        else
            error('JRI:freezeColors:checkArgs:unrecognizedOption',...
                'Unrecognized option (%s). Only ''nancolor'' is valid.',args{end-1})
        end
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

% load_colormap
function cmap = load_colormap

cmap = [0.4000 0.6000 1.0000; 0.4000 0.6000 1.0000; 0.4000 0.6000 1.0000; 0.4000 0.6000 1.0000; 0.4000 0.6000 1.0000; ...
     0.4000 0.6000 1.0000; 0.4000 0.6000 1.0000; 0.4000 0.6000 1.0000; 0.7294 0.8314 0.9569; 0.7294 0.8314 0.9569; ...
     0.7294 0.8314 0.9569; 0.7294 0.8314 0.9569; 0.7294 0.8314 0.9569; 0.7294 0.8314 0.9569; 0.7294 0.8314 0.9569; ...
     0.8039 0.8784 0.9686; 0.8039 0.8784 0.9686; 0.8039 0.8784 0.9686; 0.8039 0.8784 0.9686; 0.8039 0.8784 0.9686; ...
     0.8039 0.8784 0.9686; 0.8039 0.8784 0.9686; 0.9725 0.9725 0.9725; 0.9725 0.9725 0.9725; 0.9725 0.9725 0.9725; ...
     0.9725 0.9725 0.9725; 0.9725 0.9725 0.9725; 0.9725 0.9725 0.9725; 0.9333 0.9333 0.9333; 0.9278 0.9278 0.9278; ...
     0.9224 0.9224 0.9224; 0.9169 0.9169 0.9169; 0.9114 0.9114 0.9114; 0.9059 0.9059 0.9059; 0.8627 0.8627 0.8627; ...
     0.8627 0.8627 0.8627; 0.8627 0.8627 0.8627; 0.8627 0.8627 0.8627; 0.8627 0.8627 0.8627; 0.8627 0.8627 0.8627; ...
     0.8000 0.8000 0.8000; 0.8000 0.8000 0.8000; 0.8000 0.8000 0.8000; 0.8000 0.8000 0.8000; 0.8000 0.8000 0.8000; ...
     0.8000 0.8000 0.8000; 1.0000 0.6000 0.6000; 1.0000 0.6000 0.6000; 1.0000 0.6000 0.6000; 1.0000 0.6000 0.6000; ...
     1.0000 0.6000 0.6000; 1.0000 0.6000 0.6000; 1.0000 0.4000 0.4000; 1.0000 0.4000 0.4000; 1.0000 0.4000 0.4000; ...
     1.0000 0.4000 0.4000; 1.0000 0.4000 0.4000; 1.0000 0.4000 0.4000; 1.0000 0 0; 1.0000 0 0; ...
     1.0000 0 0; 1.0000 0 0; 1.0000 0 0; 1.0000 0 0];
end

function [cs,index] = sort_nat(c,mode)
%sort_nat: Natural order sort of cell array of strings.
% usage:  [S,INDEX] = sort_nat(C)
%
% where,
%    C is a cell array (vector) of strings to be sorted.
%    S is C, sorted in natural order.
%    INDEX is the sort order such that S = C(INDEX);
%
% Natural order sorting sorts strings containing digits in a way such that
% the numerical value of the digits is taken into account.  It is
% especially useful for sorting file names containing index numbers with
% different numbers of digits.  Often, people will use leading zeros to get
% the right sort order, but with this function you don't have to do that.
% For example, if C = {'file1.txt','file2.txt','file10.txt'}, a normal sort
% will give you
%
%       {'file1.txt'  'file10.txt'  'file2.txt'}
%
% whereas, sort_nat will give you
%
%       {'file1.txt'  'file2.txt'  'file10.txt'}
%
% See also: sort

    % Set default value for mode if necessary.
    if nargin < 2
        mode = 'ascend';
    end

    % Make sure mode is either 'ascend' or 'descend'.
    modes = strcmpi(mode,{'ascend','descend'});
    is_descend = modes(2);
    if ~any(modes)
        error('sort_nat:sortDirection',...
            'sorting direction must be ''ascend'' or ''descend''.')
    end

    % Replace runs of digits with '0'.
    c2 = regexprep(c,'\d+','0');

    % Compute char version of c2 and locations of zeros.
    s1 = char(c2);
    z = s1 == '0';

    % Extract the runs of digits and their start and end indices.
    [digruns,first,last] = regexp(c,'\d+','match','start','end');

    % Create matrix of numerical values of runs of digits and a matrix of the
    % number of digits in each run.
    num_str = length(c);
    max_len = size(s1,2);
    num_val = NaN(num_str,max_len);
    num_dig = NaN(num_str,max_len);
    for i = 1:num_str
        num_val(i,z(i,:)) = sscanf(sprintf('%s ',digruns{i}{:}),'%f');
        num_dig(i,z(i,:)) = last{i} - first{i} + 1;
    end

    % Find columns that have at least one non-NaN.  Make sure activecols is a
    % 1-by-n vector even if n = 0.
    activecols = reshape(find(~all(isnan(num_val))),1,[]);
    n = length(activecols);

    % Compute which columns in the composite matrix get the numbers.
    numcols = activecols + (1:2:2*n);

    % Compute which columns in the composite matrix get the number of digits.
    ndigcols = numcols + 1;

    % Compute which columns in the composite matrix get chars.
    charcols = true(1,max_len + 2*n);
    charcols(numcols) = false;
    charcols(ndigcols) = false;

    % Create and fill composite matrix, comp.
    comp = zeros(num_str,max_len + 2*n);
    comp(:,charcols) = double(s1);
    comp(:,numcols) = num_val(:,activecols);
    comp(:,ndigcols) = num_dig(:,activecols);

    % Sort rows of composite matrix and use index to sort c in ascending or
    % descending order, depending on mode.
    [unused,index] = sortrows(comp);
    if is_descend
        index = index(end:-1:1);
    end
    index = reshape(index,size(c));
    cs = c(index);

end
