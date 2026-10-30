all:
	bash run_all.sh

clean:
	rm -rf models cache reports/logs reports/*.png reports/*.csv reports/*.txt reports/*.npy reports/*.json reports/external_done arogya.pt ood.npz cfg.json noise.png white.png
