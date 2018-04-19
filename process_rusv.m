%read files
%pwd
%fprintf(pwd)
%cd 305/
%cd Lab\\Aurelio_Thesis_Folder\\305
%pwd

% Zoom into images, create 64x64 image, from 20khz to 84khz
% from the original images
% decrease the range to capture 22khz calls and test with real data
% Add it to GiHhub account
% https://www.gnu.org/software/octave/

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
% Main Program 
function main(varargin)

curr_dir = pwd;
%addpath(curr_dir);
%cd varargin(1);
path_wave_dir = varargin{1}{1};

%fprintf('%s\n',path_wave_dir);
%move to wave directory
%cd (path_wave_dir);
%grab all the files inside the directory
wave_files = dir (path_wave_dir);
%wave_files = wave_files(~cellfun('isempty',(regexp(wave_files.name,'\.wav'))))
wave_files = wave_files(~ismember({wave_files.name},{'.','..'}));
wave_files.name

%cd(curr_dir)
%wave_files.name

%for i=1:numel(wave_files)
%
%fullfile(path_wave_dir, wave_files(i).name)
%end

%fullfile(dir (path_wave_dir,dir(path_wave_dir))

%dir (path_wave_dir, '*.wav')

%wavefiles =  fullfile(dir (path_wave_dir, '*.wav'));
%wavefiles

%fprintf('%s',wavefiles(1).name);

%ask for path to dir
%dir_path = char('305\\');
%fprintf("dir path %s",dir_path);

%Contains wave files
%wavefiles.name

%wavefile = char(wavefiles(1).name);

%write a forloop for every file
%compute_usv(dir_path,wavefile);

%wavefiles_2 = dir **/*.WAV
%fprinf(wavefile_1)
% show_syllables
end
function compute_usv(usvdir,usv_audio_file_name)
 % FFT
 Nfft=512;
 % Gammatone filterbank
 NbChannels=64;
 fsMin=90000;
 fs=250000;
 frame_shift_ms = 0.0004*4; %ms
 frame_win_ms = 0.002;

 frame_shift=floor(frame_shift_ms*fs);
 frame_win=floor(frame_win_ms*fs);
 segment_duration = 10;

 %compute FFT of the audio
 count = 1;
 init_waitbar=1;



 %fprintf("usvdir %s %s\n", usvdir,usv_audio_file_name);



 audio_file_path = fullfile(usvdir, usv_audio_file_name);
 %fprintf('file_path %s \n',audio_file_path);

 info = audioinfo(audio_file_path);
 %{
 [~,info.SampleRate] = audioread(audio_file_path);
 fprintf('after reading sample rate\n');
 info.TotalSamplesArray = audioread(audio_file_path,'size');

 fprintf('after reading total samples array\n');
 info.TotalSamples = info.TotalSamplesArray(1);



 %}
 segment_samples = floor(info.SampleRate * segment_duration);
 %fprintf('seg samples %f\n', segment_samples);

 number_of_segments = ceil(info.TotalSamples/segment_samples);
 %fprintf('number of segments %f\n', number_of_segments);

 segment_start_rewind=0;
 frame_start=1;

 resample_normal=true;
 try
    tmp=resample(audio_segment(1:min(100,info.TotalSamples)),fs,info.SampleRate);
 catch
    resample_normal=false;
 end

 syllable_data=[];
  colormap  jet;
 for segmentID = 1:number_of_segments
     fprintf('segment id %d \n', segmentID);
     audio_start=(segmentID-1)*segment_samples+1-segment_start_rewind;
     audio_end=min(segmentID*segment_samples,info.TotalSamples);
     if exist('audioread')
        audio_segment=audioread(audio_file_path,[audio_start audio_end]);
     else
        audio_segment=AUDIOREAD(audio_file_path,[audio_start audio_end]);
     end
     if info.SampleRate < fsMin
         errordlg(sprintf('Sampling rate of audio file %s is too low (<%i kHz). \nPlease delete audio file in directory to proceed.', fname, fsOrig),'Audio file sampling rate too low');
         continue;
     end
     [syllable_data_segment,segment_start_rewind,~]=compute_musv_segment(frame_shift_ms,frame_win_ms,audio_segment,info.SampleRate,frame_start,audio_file_path,resample_normal);
     %fprintf('after computing msuv seg\n');
     frame_start=frame_start;
     if ~isempty(syllable_data_segment)
         pwd;
         syllable_data=[syllable_data syllable_data_segment];
         %img_filename = sprintf(' %d.png',segmentID);
         %fprintf(img_filename);
         %saveas(syllable_data, img_filename);

     end
 end

end

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


    %fprintf("processing Images \n ");
    % fft figure
    axes(handles.syllable_axes_fft);
    syllable_patch_fft_dB=10*log10(abs(syllable_patch_fft(1:2:end,:)+1e-5)); % in dB
    fft_range_db1=-30;
    fft_range_db1_min=-30;
    fft_range_db2=0;
    fft_peak_db=handles.syllable_stats{12,syllable_ndx};
    fft_range=[fft_range_db1_min , fft_peak_db+fft_range_db2];
    save_image=imagesc(syllable_patch_fft_dB,fft_range); axis xy; colorbar;
    colormap default;
    %colormap pink; colormap(flipud(colormap));
    colormap jet; %colormap(flipud(colormap));


     %----------writing syllable to png file ------------%

    [pathstr, name, ext] = fileparts(handles.filename)
    %hard code path
    %true_path_name = '\\\\Client\\C$\\Users\\Aurelio\\Documents\\MATLAB\\mupet\\images\';
    %path_name = pwd;
    img_filename = sprintf('%s_%d.png',name, syllable_ndx);

    %fprintf(img_filename);
    %saveas(gca, img_filename);


    %-----------end of writing image to file ------------%

    set(gca,'YTick',[0:size(syllable_patch_fft_dB,1)/5:size(syllable_patch_fft_dB,1)]) % FFT bands
    set(gca,'YTickLabel',fix([0:handles.sample_frequency/2/5:handles.sample_frequency/2]/1e3)) % FFT bands
    set(gca,'XTick',[0:syllable_patch_window/6:syllable_patch_window]) % frequency
    set(gca,'XTickLabel',fix([0:handles.frame_shift_ms*syllable_patch_window/6:syllable_patch_window*handles.frame_shift_ms]*1e3)) % frequency
    set(gca, 'FontSize',handles.FontSize3,'FontName','default');
    xlabel('Time (milliseconds)','FontSize',handles.FontSize2,'FontName','default');
    ylabel('Frequency [kHz]','FontSize',handles.FontSize2,'FontName','default');
    title('Sonogram','FontSize',handles.FontSize1,'FontName','default','FontWeight','bold');
    %ylim([size(syllable_fft,1)/125000*25000/2 size(syllable_fft,1)/2]);
    ylim([size(syllable_fft,1)/100000*30000/2 size(syllable_fft,1)/2]);





    axvals=axis; %text(axvals(2)/2,axvals(4)/2,{'MUPET version 1.0', '(unreleased)'},'Color',[0.9 0.9 0.9],'FontSize',handles.FontSize1+10,'HorizontalAlignment','center','Rotation',45);

    % gt figure
    axes(handles.syllable_axes_gt);
    imagesc(syllable_patch_gt, [0, max(max(syllable_patch_gt))]); axis xy; colorbar;
    %colormap default;
    %colormap pink; colormap(flipud(colormap));
    set(gca,'YTick',[0:size(syllable_patch_gt,1)/4:size(syllable_patch_gt,1)]) % GT bands
    set(gca,'XTick',[0:syllable_patch_window/6:syllable_patch_window]) % frequency
    set(gca,'XTickLabel',fix([0:handles.frame_shift_ms*syllable_patch_window/6:syllable_patch_window*handles.frame_shift_ms]*1e3)) % frequency
    set(gca, 'FontSize',handles.FontSize3,'FontName','default');
    xlabel('Time (milliseconds)','FontSize',handles.FontSize2,'FontName','default');
    ylabel('Gammatone bands','FontSize',handles.FontSize2,'FontName','default');
    title('Gammatone representation','FontSize',handles.FontSize1,'FontName','default','fontweight','bold');
    axvals=axis; %text(axvals(2)/2,axvals(4)/2,{'MUPET version 1.0', '(unreleased)'},'Color',[0.9 0.9 0.9],'FontSize',handles.FontSize1+10,'HorizontalAlignment','center','Rotation',45);

    % display syllable information
    syllable_info_string1=sprintf('%s%s%s%s%s', ...
        sprintf('File: %s\n',handles.filename), ...
        sprintf('Number of syllables in file: %i\n', size(handles.syllable_stats,2)), ...
        sprintf('Syllable number shown: %i of %i\n',syllable_ndx, size(handles.syllable_stats,2)), ...
        sprintf('Syllable start time: %.4f sec\n',handles.syllable_stats{8,syllable_ndx}), ...
        sprintf('Syllable end time: %.4f sec\n',handles.syllable_stats{9,syllable_ndx}));
    set(handles.syllable_info_panel1, 'HorizontalAlignment', 'left');
    set(handles.syllable_info_panel1, 'FontSize',handles.FontSize2+1,'FontName','default');
    set(handles.syllable_info_panel1, 'FontAngle', 'normal');
    set(handles.syllable_info_panel1, 'string', syllable_info_string1);

    syllable_info_string2=sprintf('%s%s%s%s%s', ...
        sprintf('  starting frequency: %.2f kHz\n', handles.syllable_stats{2,syllable_ndx}), ...
        sprintf('  final frequency: %.2f kHz\n', handles.syllable_stats{3,syllable_ndx}), ...
        sprintf('  minimum frequency: %.2f kHz\n', handles.syllable_stats{4,syllable_ndx}), ...
        sprintf('  maximum frequency: %.2f kHz\n', handles.syllable_stats{5,syllable_ndx}), ...
        sprintf('  mean frequency: %.2f kHz\n',handles.syllable_stats{10,syllable_ndx}));
    set(handles.syllable_info_panel2, 'HorizontalAlignment', 'left');
    set(handles.syllable_info_panel2, 'FontSize',handles.FontSize2+1,'FontName','default');
    set(handles.syllable_info_panel2, 'FontAngle', 'normal');
    set(handles.syllable_info_panel2, 'string', syllable_info_string2);

    if handles.syllable_stats{14,syllable_ndx} == -100
        inter_syllable_interval='_';
    else
        if handles.syllable_stats{14,syllable_ndx}>=1e3
            inter_syllable_interval=sprintf('%.4f sec',handles.syllable_stats{14,syllable_ndx}/1e3);
        else
            inter_syllable_interval=sprintf('%.2f ms',handles.syllable_stats{14,syllable_ndx});
        end
    end

    % display syllable information
    syllable_info_string3=sprintf('%s%s%s%s%s%s', ...
        sprintf('  frequency bandwidth: %.2f kHz\n',handles.syllable_stats{6,syllable_ndx}), ...
        sprintf('  syllable duration: %.2f ms\n', handles.syllable_stats{13,syllable_ndx}), ...
        sprintf('  inter-syllable interval: %s\n',inter_syllable_interval), ...
        sprintf('  total syllable energy: %.2f dB\n', handles.syllable_stats{11,syllable_ndx}), ...
        sprintf('  peak syllable amplitude: %.2f dB\n', handles.syllable_stats{12,syllable_ndx}));
    set(handles.syllable_info_panel3, 'HorizontalAlignment', 'left');
    set(handles.syllable_info_panel3, 'FontSize',handles.FontSize2+1,'FontName','default');
    set(handles.syllable_info_panel3, 'FontAngle', 'normal');
    set(handles.syllable_info_panel3, 'string', syllable_info_string3);

end


% compute_musv
function [syllable_data, syllable_stats, filestats, fs] = compute_musv(datadir,flist,handles,existonly)

    % FFT
    Nfft=256*2;

    if ~exist('existonly', 'var')
        existonly=false;
    end

    % Gammatone filterbank
    NbChannels=64;
    fsMin=90000;
    fs=250000;
    frame_shift=floor(handles.frame_shift_ms*fs);
    frame_win=floor(handles.frame_win_ms*fs);
    gtdir=fullfile(handles.audiodir);
    if ~exist(gtdir,'dir')
      mkdir(gtdir)
    end
    %GTB=gammatone_matrix_sigmoid(Nfft*2,fs,NbChannels);

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
                 end
             end
             TotNbFrames=floor((info.TotalSamples/info.SampleRate*fs-frame_win+frame_shift)/frame_shift);

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


% process_file
function handles=process_file(handles)

    wav_items=get(handles.wav_list,'string');
    selected_wav=get(handles.wav_list,'value');
    wav_dir=get(handles.wav_directory,'string');
    if ~isempty(wav_dir)
        [syllable_data, syllable_stats, filestats, fs]=compute_musv(wav_dir,wav_items(selected_wav),handles);
        handles.syllable_stats = syllable_stats;
        handles.syllable_data = syllable_data;
        handles.sample_frequency = fs;
        handles.filename = wav_items{selected_wav};
        nb_syllables=filestats.nb_of_syllables;
        if nb_syllables >= 1
            set(handles.syllable_slider,'Value',0);
            if nb_syllables==1
                set(handles.syllable_slider,'Visible','off');
            else
                set(handles.syllable_slider,'Visible','on');
                if nb_syllables==2
                    set(handles.syllable_slider,'Max',1);
                else
                    set(handles.syllable_slider,'Max',nb_syllables-2);
                end
                set(handles.syllable_slider,'Value',0);
                set(handles.syllable_slider,'Min',0);
                set(handles.syllable_slider,'SliderStep',[1/(double(nb_syllables-1)) 1/(double(nb_syllables-1))]);
            end
            syllable_ndx=1;

            % make syllable patch
            show_syllables(handles,syllable_ndx);
        else
            errordlg(sprintf(' ***              No syllables found in file              *** '),'MUPET info');
        end
    end

end

function [syllable_data,segment_start_rewind,nb_of_syllables] = compute_musv_segment(frame_shift_ms,frame_win_ms,audio_segment,fsOrig,frame_start,audiofile,resample_normal)

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
smooth_fac=floor(5*(0.0016/frame_shift_ms));
smooth_fac_low=10;
grow_fac=floor(3*(0.0016/frame_shift_ms));

% frequency
fsMin=90000;
fs=250000;
frame_shift=floor(frame_shift_ms*fs);
frame_win=floor(frame_win_ms*fs);

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
fmin=30000
%fmax=110000;
fmax=90000
Nmin=floor(Nfft/(fs/2)*fmin);
Nmax=floor(Nfft/(fs/2)*fmax);
[ sonogram, E_low, E_usv, T]=FE_GT_spectra(audio_segment, fs, frame_win, frame_shift, Nfft, Nmin, Nmax);

% Gaussian noise floor estimation by median
logE = log(E_usv);
logE_noise = median(logE);
logE_norm = logE - logE_noise;
logE_low = log(E_low);
logE_low_noise = median(logE_low);
logE_low_norm = logE_low - logE_low_noise;

denoising = true;
% syllable activity detection
if denoising
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
%nbFrames=size(gt_sonogram,2);

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
%noise_floor_GT=median(gt_sonogram,2);
noise_floor_X=median(sonogram,2);



%GT = gt_sonogram - repmat(noise_floor_GT, 1, size(gt_sonogram,2));
X = sonogram - repmat(noise_floor_X, 1, size(sonogram,2));
% X=sonogram;


% spectral noise floor
%[counts,vals]=hist(GT(:),100);
%[sigma_noise, mu_noise] = gaussfit( vals, counts./sum(counts), handles);

% syllable selection
syllable_data=cell(6,length(end_sad));

nb_of_syllables=length(end_sad);
for k=1:nb_of_syllables
    syllable_data{1,k}=audiofile;
    %syllable_data{2,k}=GT(:,start_sad(k):end_sad(k)-1); % gt
    syllable_data{3,k}=X(:,start_sad(k):end_sad(k)-1); % fft
    syllable_data{4,k}=E_usv(:,start_sad(k):end_sad(k)-1)*(Nmax-Nmin); % energy
    syllable_data{5,k}=frame_start+start_sad(k)-1;
    syllable_data{6,k}=frame_start+end_sad(k)-1;
    syllable_data{7,k}=1;    % syllable considered for analysis
    %syllable_data{8,k}=sigma_noise;
    %syllable_data{9,k}=mu_noise;
end

syllable_onset=cell2mat(syllable_data(5,:));
syllable_offset=cell2mat(syllable_data(6,:));
syllable_distance = [syllable_onset(2:end) - syllable_offset(1:end-1) 0];
syllables_to_remove = [];
min_distance=5.0;

for k=1:nb_of_syllables
%     fprintf('%.4f %.2f %.2f\n',syllable_data{5,k}*frame_shift/fs, (syllable_data{6,k}-syllable_data{5,k})*frame_shift/fs*1e3 ,syllable_distance(k)*frame_shift/fs*1e3);
    if syllable_distance(k)*frame_shift/fs*1e3 < min_distance && syllable_distance(k)*frame_shift/fs*1e3 > 0
        %syllable_data{2,k+1}=GT(:,start_sad(k):end_sad(k+1)-1);
        syllable_data{3,k+1}=X(:,start_sad(k):end_sad(k+1)-1); % fft
        syllable_data{4,k+1}=E_usv(:,start_sad(k):end_sad(k+1)-1)*(Nmax-Nmin); % energy
        syllable_data{4,k+1}=E_usv(:,start_sad(k):end_sad(k+1)-1)*(Nmax-Nmin); % energy
        syllable_data{5,k+1}=syllable_data{5,k};
        syllable_data{6,k+1}=syllable_data{6,k+1};
        syllable_data{7,k+1}=1;    % syllable considered for analysis
        %syllable_data{8,k+1}= syllable_data{8,k};
        %syllable_data{9,k+1}=syllable_data{9,k};
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
    %syllable_constant(k)=sum(syllable_data{2,k}(:,1)-syllable_data{2,k}(:,end))~=0;
end

% remove constant patches
syllable_selection = syllable_constant ;
syllable_data(:,~syllable_selection)=[];
nb_of_syllables=size(syllable_data,2);

end

% FE_GT_spectra
function [Xf,E_low, E_usv,T]=FE_GT_spectra(sam,fs,FrameLen,FrameShift,Nfft,Nmin,Nmax,W,M)

    GTfloor=1e-3;

    %if ~exist('W', 'var')
    %    W=gammatone_matrix_sigmoid(Nfft*2,fs);
    %end
   % if ~exist('M', 'var')
    %    M=1;
    %end

    % truncate as in ReadWave
    NbFr=floor( (length(sam)-FrameLen+FrameShift)/FrameShift);
    sam=sam(1:NbFr*FrameShift+FrameLen-FrameShift);

    % % Low pass removal
    % fmax=0.99*fs/2;
    % sam=[0 reshape(sam,1,length(sam))];
    % T=length(sam);

    % Low pass removal
    fmin=25000;
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
    %lp=max(W*Xf_hp,GTfloor);
    %lp=log(lp);
    %lp=arma_filtering(lp,M);

end
