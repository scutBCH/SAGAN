set emb_path_prefix=C:\Users\12425\Desktop\SAGAN\data
set adj_path_prefix=C:\Users\12425\Desktop\SAGAN\data\adj
set adj_suffix=_20w_n50_matrix


set src_language=en
set tgt_language=ru
set exp_name_prefix=en_ru
for /l %%i in (1,1,10) do python unsupervised.py  --exp_name adv --exp_id %exp_name_prefix%_time%%i  --src_lang  %src_language%  --tgt_lang %tgt_language% --src_emb %emb_path_prefix%\\wiki.%src_language%.vec --tgt_emb %emb_path_prefix%\\wiki.%tgt_language%.vec  --adj_a %adj_path_prefix%\%src_language%%adj_suffix% --adj_b %adj_path_prefix%\%tgt_language%%adj_suffix%


set src_language=en
set tgt_language=es
set exp_name_prefix=en_es
for /l %%i in (1,1,10) do python unsupervised.py  --exp_name adv --exp_id %exp_name_prefix%_time%%i  --src_lang  %src_language%  --tgt_lang %tgt_language% --src_emb %emb_path_prefix%\\wiki.%src_language%.vec --tgt_emb %emb_path_prefix%\\wiki.%tgt_language%.vec  --adj_a %adj_path_prefix%\%src_language%%adj_suffix% --adj_b %adj_path_prefix%\%tgt_language%%adj_suffix%





