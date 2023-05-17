import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

uploaded_file = st.sidebar.file_uploader(label="", type=['csv'])

sidebar_col1,sidebar_col2 = st.sidebar.columns((125,125))

freq_slider = sidebar_col1.slider("Frequency", 1, 10, 1, 1, key='freq')
amp_slider = sidebar_col1.slider("Amplitude", 1, 10, 1, 1, key='amp')
noise_checkbox=sidebar_col2.checkbox(label='Noise')
snr = sidebar_col1.slider("SNR",value=50,min_value=0,max_value=50,step=5 )



if uploaded_file is not None:
    sampling_csv = sidebar_col2.checkbox('Sample')
    reconstruct_csv=sidebar_col2.checkbox("Reconstruct", key="reconstruct_signal_checkbox")

    sampling_freq_csv=sidebar_col2.slider(label="Sampling Frequency (HZ)",min_value=float(1),max_value=float(5),value=float(5))
    df = pd.read_csv(uploaded_file)


def csv_sampling(dataframe): 
    frequency=sampling_freq_csv
    period=1/frequency
    no_cycles=dataframe.iloc[:,0].max()/period
    nyquist_freq=2*frequency
    no_points=dataframe.shape[0]
    points_per_cycle=no_points/no_cycles
    step=points_per_cycle/nyquist_freq
    sampling_time=[]
    sampling_amplitude=[]
    for i in range(int(step/2), int(no_points), int(step)):
        sampling_time.append(dataframe.iloc[i, 0])
        sampling_amplitude.append(dataframe.iloc[i, 1])
    global sampling_points
    sampling_points=pd.DataFrame({"time": sampling_time, "amplitude": sampling_amplitude})
    return sampling_points

def csv_interpolation(df, sampling_points):
 time = df['time']
 amplitude = df['amplitude']
 sampled_time = sampling_points['time']
 sampled_amplitude = sampling_points['amplitude']
 T = (sampled_time[1] - sampled_time[0])
 sincM = np.tile(time, (len(sampled_time), 1)).T - np.tile(sampled_time, (len(time), 1))
 yNew = np.dot(np.sinc(sincM/T), sampled_amplitude)
 reconstructed_signal = pd.DataFrame({"time": time, "amplitude": yNew})
 difference = amplitude - reconstructed_signal["amplitude"]
 return reconstructed_signal, difference

def csv_plot(df):
    amplitude = df['amplitude']
    time = df['time']
    dataframe_noise=pd.DataFrame({"time": time, "amplitude": amplitude})
    fig,ax= plt.subplots()

    if noise_checkbox:
        power=df['amplitude']**2
        signal_average_power=np.mean(power)
        signal_averagePower_db=10*np.log10(signal_average_power)
        noise_db=signal_averagePower_db-snr
        noise_watts=10**(noise_db/10)
        mean_noise=0
        noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(df['amplitude']))
        noise_signal=df['amplitude']+noise
        dataframe_noise=pd.DataFrame({"time": time, "amplitude": noise_signal})
        ax.plot(time, noise_signal,color='black' ,label="Original Signal")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_xlim([0, 3])
        ax.set_ylim([-1, 1])
        if sampling_csv==0:
            ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
    else:
        ax.plot(time, amplitude,color='black' ,label="Original Signal")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_xlim([0, 3])
        ax.set_ylim([-1, 1])
        if sampling_csv==0:
            ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))


    if sampling_csv==1:
          sampling_points=csv_sampling(dataframe_noise)
          
          ax.plot(sampling_points['time'], sampling_points['amplitude'],'ro',label="Sampling Points")
          ax.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))


    
    if reconstruct_csv:
     fig,ax= plt.subplots(ncols=2,nrows=2)
     fig_diff,ax_diff= plt.subplots()
     fig.subplots_adjust(wspace=0.4, hspace=0.4)
     
     reconstructed_signal, difference = csv_interpolation(dataframe_noise, sampling_points)
     ax[0,1].plot(reconstructed_signal['time'], reconstructed_signal['amplitude'], color='orange', label='Reconstructed Signal')
     ax[0,1].set_xlabel('Time')
     ax[0,1].set_ylabel('Amplitude')
     ax[0,1].set_xlim([0, 3])
     ax[0,1].set_ylim([-0.5, 0.5])
     ax[0,1].legend(loc='upper right', fontsize=8.5, bbox_to_anchor=(1.1, 1.05))

     if noise_checkbox:
      ax[0,0].plot(time, noise_signal,color='black' ,label="Original Signal")
      ax[0,0].set_xlabel("Time")
      ax[0,0].set_ylabel("Amplitude")
      ax[0,0].set_xlim([0, 3])
      ax[0,0].set_ylim([-0.5, 0.5])
      ax[0,0].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
      if sampling_csv==1:
          sampling_points=csv_sampling(dataframe_noise)
          ax[0,0].plot(sampling_points['time'], sampling_points['amplitude'],'ro',label="Sampling Points")
          ax[0,0].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
      ax_diff.plot(reconstructed_signal['time'], difference, label='Difference')
      ax_diff.set_xlabel('Time')
      ax_diff.set_ylabel('Amplitude')
      ax_diff.set_xlim([0, 3])
      ax_diff.set_ylim([-1, 1])
      ax_diff.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
      ax[1,0].remove()
      ax[1,1].remove()
        

     else:
         ax[0,0].plot(time, amplitude,color='black' ,label="Original Signal")
         ax[0,0].set_xlabel("Time")
         ax[0,0].set_ylabel("Amplitude")
         ax[0,0].set_xlim([0, 3])
         ax[0,0].set_ylim([-0.5, 0.5])
         ax[0,0].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
         if sampling_csv==1:
          sampling_points=csv_sampling(dataframe_noise)
          ax[0,0].plot(sampling_points['time'], sampling_points['amplitude'],'ro',label="Sampling Points")
          ax[0,0].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
         ax_diff.plot(reconstructed_signal['time'], difference, label='Difference')
         ax_diff.set_xlabel('Time')
         ax_diff.set_ylabel('Amplitude')
         ax_diff.set_xlim([0, 3])
         ax_diff.set_ylim([-1, 1])
         ax_diff.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
         ax[1,0].remove()
         ax[1,1].remove()

     st.pyplot(fig)
     st.pyplot(fig_diff)
    else:
        st.pyplot(fig)

try:
    
    csv_plot(df)

except Exception as e:
    print(e)


def composer_sampling(fsample, t, sin):
    time_range = (max(t) - min(t))
    samp_rate = int((len(t) / time_range) / fsample)
    if samp_rate == 0:
        samp_rate = 1
    
    samp_time = t[::samp_rate]
    samp_amp = sin[::samp_rate]
    
    # Interpolate to match length of samp_time
    interp_fn = interp1d(samp_time, samp_amp, kind='linear', fill_value='extrapolate')
    samp_amp = interp_fn(samp_time)
    
    return samp_time, samp_amp

def composer_interpolation(nt_array, sampled_amplitude, time):
    if len(nt_array) != len(sampled_amplitude):
        raise Exception('x and s must be the same length')
    T = (sampled_amplitude[1] - sampled_amplitude[0])
    sincM = np.tile(time, (len(sampled_amplitude), 1)).T - np.tile(sampled_amplitude, (len(time), 1))
    y_new = np.dot(np.sinc(sincM/T), nt_array.T)
    return y_new

if 'noise' not in st.session_state:
    st.session_state.noise = False

def plot_signals(signals):
    if (sampling==0 and reconstruct==0)or(sampling==1 and reconstruct==0) or (sampling==0 and reconstruct==1):
     fig, ax = plt.subplots(ncols=2,figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})
     fig_diff,ax_diff=plt.subplots()
    else:
        fig, ax = plt.subplots(ncols=2,figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1]})
        fig_diff,ax_diff=plt.subplots()

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Amplitude")
    ax_diff.set_xlabel("Time")
    ax_diff.set_ylabel("Amplitude")

    samp_time = []
    samp_amp = []

    for i, (freq, amp) in enumerate(signals):
        num_periods = freq
        x = np.linspace(0, num_periods/freq, 100*num_periods)
        y = amp * np.sin(2 * np.pi * freq * x)
        
        #line, = ax[0].plot(x, y, label=f"Signal {i+1}")

    if signals:
        sum_y = np.zeros_like(y)
        for freq, amp in signals:
            sum_y += amp * np.sin(2 * np.pi * freq * x)
        
        if st.session_state.noise:
            snr = st.session_state.snr
            signal_power = np.sum(sum_y ** 2) / len(sum_y)
            noise_power = signal_power / (10 ** (snr/10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(sum_y))
            sum_y += noise
        sum_line, = ax[0].plot(x, sum_y, label="Signal Sum")

        if st.session_state.reconstruct_signal:
            interpolated_time = np.linspace(0, max(x), freq*100)
            if sampling:
                samp_freq = sampling_freq
                samp_time, samp_amp = composer_sampling(samp_freq, x, sum_y)
                y_new = composer_interpolation(samp_amp, samp_time, interpolated_time)
                ax[1].plot(interpolated_time, y_new, 'orange', label='Reconstructed Signal')
                ax[0].plot(samp_time, samp_amp, 'ro', label='Sampled points')
                ax[0].legend(loc="upper right", fontsize=20, bbox_to_anchor=(1.1, 1.05))
                difference=sum_y-y_new
                ax_diff.plot(difference, 'blue', label='Difference')
                ax_diff.legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
            else:
                ax[1].remove()
                ax_diff.remove()
            ax[1].legend(loc="upper right", fontsize=20, bbox_to_anchor=(1.1, 1.05))
        elif sampling:
            samp_freq = sampling_freq
            samp_time, samp_amp = composer_sampling(samp_freq, x, sum_y)
            ax[0].plot(samp_time, samp_amp, 'ro', label='Sampled points')
            ax[0].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
            ax[1].remove()
            ax_diff.remove()

        else:
            samp_time = x
            samp_amp = sum_y
            ax[0].legend(loc="upper right", fontsize=8.5, bbox_to_anchor=(1.1, 1.05))
            ax[1].remove()
            ax_diff.remove()
    return fig,fig_diff


if "signals" not in st.session_state:
    st.session_state.signals = []
    st.session_state.reconstruct_signal = False
sampling = st.session_state.get("sampling", False)


# Calculate the maximum frequency from the currently drawn signals
if st.session_state.signals:
    max_freq = max([freq for freq, amp in st.session_state.signals])
else:
    max_freq = 10
    

if uploaded_file is None:
    sampling= sidebar_col2.checkbox('Sample')
    reconstruct=sidebar_col2.checkbox("Reconstruct", key="reconstruct_signal_checkbox")


if st.session_state.signals:
    max_freq = max([freq for freq, amp in st.session_state.signals])
else:
    max_freq = 10

if noise_checkbox:
    st.session_state.snr = snr
    st.session_state.noise = True
else:
    st.session_state.noise = False

if "reconstruct_signal_checkbox" in st.session_state:
    st.session_state.reconstruct_signal = st.session_state.reconstruct_signal_checkbox

if sidebar_col1.button("Add"):
    st.session_state.signals.append((freq_slider, amp_slider))

if st.session_state.signals:
    max_freq = max([freq for freq, amp in st.session_state.signals])
    sampling_options=["actual","normalized"]
    sampling_chosen=sidebar_col2.selectbox("Sampling Frequency",sampling_options)

    if sampling_chosen=="actual":
     sampling_freq_slider_actual = sidebar_col2.slider("Sampling Frequency (Hz)", min_value=float(0.5*max_freq), max_value=float(4*max_freq), step=float(1))
     sampling_freq=sampling_freq_slider_actual
    elif sampling_chosen=="normalized":
        sampling_freq_slider_normalized=sidebar_col2.slider("Sampling Frequency (Fmax)", min_value=1, max_value=20, step=1)
        sampling_freq=sampling_freq_slider_normalized*max_freq

    signal_options = [f"Signal {i+1}" for i in range(len(st.session_state.signals))]
    signal_to_delete = sidebar_col2.selectbox("Signal to delete", signal_options)
    if sidebar_col1.button("Delete"):
        signal_index = signal_options.index(signal_to_delete)
        st.session_state.signals.pop(signal_index)
        st.experimental_rerun()


if st.session_state.signals:
    plot,difference = plot_signals(st.session_state.signals)
    st.pyplot(plot)
    if reconstruct==1:
     st.pyplot(difference)