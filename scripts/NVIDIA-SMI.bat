FOR /L %%G IN (1,1,1000) DO (
	"nvidia-smi.exe"
	@ping -n 5 localhost> nul
	CLS
)
pause