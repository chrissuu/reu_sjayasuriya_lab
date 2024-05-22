function [speech_snr]= FFT_Compression(input_audio,N,L)

    % input_audio - input clean speech signal
    % N – frame size (in samples)
    % L - number of components to choose from the spectrum
    % l1 - length of the input signal
    % fs1 - Sampling frequency of the input signal
    % reconstructed_signal - concatenated signal obtained after IFFT of the spectrum
    % original_signal - concatenated original signal 
    % no_of_frames - number of frames of the signal based on the input N
    % spectrum - stores the N point FFT of the signal frame
    % diff_signal - difference between original signal and reconstructed
    %               signal
    % power_signal - stores the power of the original signal
    % power_noise - stores the power of the difference signal
    % clean_fraction - ratio of power signal and power noise
    % speech_snr - stores the SNR between the clean signal and compressed
    %              signal
    
    % input_audio = 'cleanspeech.wav';
    [s1,fs1]=audioread(input_audio);
    
    l1=length(s1);
    reconstructed_signal=[];
    original_signal=[];
    % N=256;
    % L=256;
    % Calculate the number of frames
    no_of_frames = fix(l1/N);
    
    for k = 1 : no_of_frames
        % Compute indices for current frame
        n = (1:N)+(N*(k-1));
    
        % Find FFT
        spectrum = fft(s1(n)); 
    
        % Choose L components (set L+1 to end as 0) of spectrum and force symmetry
        spectrum(L+1:end)=0;
        for j = 2:N/2
            spectrum(N+2-j) = conj(spectrum(j));
        end
    
        % Combine arrays for reconstruction
        reconstructed_signal =[reconstructed_signal;ifft(spectrum)];
        original_signal = [original_signal;s1(n)];
    
    
        % Plot spectrum
        subplot(3,1,1)
        plot(n,spectrum,'b');
        title("Spectrum of the Signal");
    
        % Clean original signal
        subplot(3,1,2)
        plot(n,s1(n),'b');
        title("Original Signal");
    
        % Reconstructed signal
        subplot(3,1,3)
        plot(n,ifft(spectrum),'r');
        title("Reconstructed Signal");
        
        
        % pause;
    
    end
    
    % SNR of clean speech signal
    diff_signal = original_signal - reconstructed_signal;
    
    power_signal = sum(abs(original_signal'*original_signal));
    power_noise = sum(abs(diff_signal'*diff_signal));
    
    clean_fraction = power_signal/power_noise;
    speech_snr = 10* log10(clean_fraction);
    
    disp("SNR =")
    disp(speech_snr)
    
    % Spectrogram of the Original and Reconstructed signal
    figure();
    subplot(2,1,1)
    spectrogram(original_signal)
    title("Original Signal Spectrogram");
    subplot(2,1,2)
    spectrogram(reconstructed_signal)
    title("Reconstructed Signal Spectrogram");
    
    % Listening to the audio
    % sound(original_signal,fs1);
    % pause
    % sound(reconstructed_signal,fs1);
end