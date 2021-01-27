
f, ax = plt.subplots()


for t_id in df['track_id'].unique()[:5]:
    df_ = df[['frame', 'volume']][(df['track_id'] == t_id) & (df['seg_id'] != 0)]

    ax.plot(df_['frame'], df_['volume'])
    
ax.set_ylim(0, 5000)
ax.set_xlabel('Frame number')
ax.set_ylabel('Volume')

output_name = Path(r'reports\figures\track_Volume_over_FrameNumber.pdf')

if not output_name.parent.is_dir():
    os.makedirs(output_name.parent)

f.savefig(output_name)