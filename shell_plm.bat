set emb_path_prefix=C:\Users\12425\Desktop\SAGAN\data


set emb_path=C:\Users\12425\Desktop\SAGAN\dumped\dis-5times-gen-1times_0.5_0.5\en_ru_time
set src_language=en
set tgt_language=ru
set exp_name_prefix=en_ru
for /l %%i in (1,1,10) do python plm_bi.py  --exp_name plm --exp_id %exp_name_prefix%_time%%i    --src_lang  %src_language%  --tgt_lang %tgt_language% --src_emb %emb_path%%%i\\mapped-vectors-%src_language%.pth --tgt_emb %emb_path%%%i\\mapped-vectors-%tgt_language%.pth


