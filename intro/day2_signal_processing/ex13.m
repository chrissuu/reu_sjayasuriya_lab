function [s,fs,bits] = ex13(infile, playstate)
    % ex13(infile,playstate)
    %
    % infile -.WAV input file
    % playstate –Switch playback on/off
    %
    % s –signal loaded from infile
    % fs –sample rate% bits –bits per sample
    %
    % Function loads infile, displaysentire
    % record, then optionally plays back the
    % sound depending upon state of playstate

    [s,fs]=audioread(infile);
    info = audioinfo(infile);
    bits = info.BitsPerSample;
%     plot(s)
%     plot(s,':')
%     plot(s,'r:')
%     plot(s(1:256))

%     j=1:256;
%     plot(s(j));

%     j=1:256;
%     plot(j+512,s(j));

    j = 1:256;
    N=256;
    j=j+N;
    plot(j,s(j));
    title('Cleanspeech time waveform');
    xlabel('Sample Number');
    ylabel('Normalized Amplitude');
    if playstate == 1
        sound(s,fs);
    end
end

