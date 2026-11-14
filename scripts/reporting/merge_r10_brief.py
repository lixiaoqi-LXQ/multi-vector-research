from pathlib import Path
import json,statistics,copy
R=Path('/home/lxq/graduate/ai-workspace/research/multivector');D=R/'report';A=R/'report-assets'
p=D/'lotte-r10-full-query-records.json';s=p.read_text();old=json.loads(s)
backup=D/'lotte-r10-full-query-records-before95merge.json'
if not backup.exists():backup.write_text(s)
data=copy.deepcopy(json.loads(backup.read_text()));new=json.loads((D/'lotte-r10-95-chunks-data.json').read_text())['chunk_results'][0]
for i,x in enumerate(data['records']):
 if x['system']=='Chimera' and x['target_recall10']==.95:
  data.setdefault('historical_failed_points',[]).append(x);n=copy.deepcopy(x);n.update(rounds=[dict(r,**{'recall@10':r['recall10'],'quality_target_met':r['recall10']>=.95}) for r in new['rounds']],mean_of_round_QPS=new['mean_QPS'],mean_of_round_mean_ms=new['mean_latency_ms'],mean_of_round_p99_ms=new['mean_round_p99_ms'],full_recall10=new['mean_recall10'],qualification=True,qualified=True,qualification_note='All2931 x3 mean Recall@10 >= .95; no heldout; bounded B/E full3 selection',point=new['point'],evidence={'source_run':new['source_run'],'source_sha256':new['source_sha256']},formal_output_accepted=True,performance_window_accepted=True)
  n.pop('pooled_latency_ms',None);n.pop('qrels',None);data['records'][i]=n
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
brief=D/'lotte-r10-brief.md';b=brief.read_text();bb=D/'lotte-r10-brief-before95merge.md'
if not bb.exists():bb.write_text(b)
b=b.replace('> 更新：0.95选参与chunks=1/4/5的全量三轮已完成，Chimera达到Recall@10=0.958649、585.25 QPS，详见[最新0.95报告](lotte-r10-95-chunks.md)。下文保留首批实验及其未达标点。\n\n','')
b=b.replace('R10≥0.95时，PLAID为54.2 QPS；Chimera冻结点的全量召回约0.9486，未达标。','R10≥0.95时，Chimera达到585.25 QPS、R10=0.958649；PLAID最快放置为54.22 QPS、R10=0.956534，前者约10.79倍。CPU20下，chunks=1快于4和默认5，继续采用1。')
b=b.replace('| Chimera | 0.95 | 710.1 | 0.948607 | **未达标** |','| Chimera | 0.95 | 585.25 | 0.958649 | 达标 |')
b=b.replace('lotte-r10-full-query-v2.png','lotte-r10-full-query-v3.png')
b=b.replace('图中误差线为三轮最小—最大值，尾延迟为各轮p99的均值；斜线保留未达标点。主配置均为20线程。PLAID两种放置在每个目标下各8,793对结果ID/FP32评分逐位一致；Chimera未达标点未用于0.95目标的倍率比较。','图中误差线为三轮最小—最大值，尾延迟为各轮p99的均值。主配置均为20线程。PLAID两种放置在每个目标下各8,793对结果ID/FP32评分逐位一致。Chimera的0.95主点已替换为新一轮全量三轮达标配置；原0.948607未达标点保留于历史记录，未参与倍率计算。\n\n0.95配置：`nprobe=128 / k_refine=1000 / k_full_bit=50 / cagra_itopk_size=160 / chunks=1`。选参使用全部2,931条查询，不划分留出集；B、E两邻点均复测三轮后，按达标点的平均QPS选E。选参单轮记录与三轮明细见[证据](lotte-r10-95-chunks-data.json)。')
extra=(D/'lotte-r10-95-chunks.md').read_text();chunk=extra.split('## 同参比较 chunks=1 / 4 / 5\n')[1].split('## 制品与执行状态')[0]
chunk=chunk.replace('## 0.95配置的瓶颈与 breakdown','### 0.95配置的瓶颈与 breakdown').replace('## UMA上可优化的空间','### UMA上可优化的空间')
b=b.replace('## 3. 补充短测：线程与chunk','## 3. 0.95配置：chunks全量比较与UMA分析\n\n'+chunk+'\n## 4. 历史补充短测：线程与chunk')
b=b.replace('本轮性能测试已结束。交付本简报后按要求停止后续实验，R100、其他数据集和消融均不继续。','Recall@10阶段已完成。按最新要求，Recall@100正在使用相同LoTTE索引继续测试；结果完成后更新本简报。其他数据集、build和消融尚未启动。')
brief.write_text(b)
# Detailed first report remains an evidence-rich companion; latest summary placed before historical body.
p=D/'lotte-r10-report.md';oldtext=p.read_text();bak=D/'lotte-r10-report-before95merge.md'
if not bak.exists():bak.write_text(oldtext)
notice='''## 当前结论（已合并0.95补测）

最新汇总统一见[第一版简报](lotte-r10-brief.md)，其中已合并0.95选参、chunks=1/4/5全量三轮、CPU/GPU breakdown与UMA优化方向。

| 目标Recall@10 | Chimera QPS / 实际Recall | PLAID最快放置QPS / 实际Recall | QPS倍率 |
|---|---|---|---:|
| 0.90 | 811.8 / 0.907870 | 169.0 / 0.925316 | 4.80 |
| 0.95 | 585.25 / 0.958649 | 54.22 / 0.956534 | 10.79 |

0.95采用 `nprobe128 / refine1000 / full50 / itopk160 / chunks1`，CPU20，全2,931条查询×3轮，无留出集。chunks4/5分别为371.38/339.35 QPS，均达标；继续采用1。32条诊断查询中，chunks4的精排量仍为50篇/查询，拆为12/13/12/13，CPU区域时间增长超过重叠收益。当前优先优化CPU任务粒度、共享小缓冲区及事件复用；DRAM带宽是否饱和尚缺有效计数。

Recall@100按最新要求继续测试。下方首批详细采证保留原始配置和未达标点，最新0.95结论以上表与简报为准。

'''
if '## 当前结论（已合并0.95补测）' not in oldtext:p.write_text(oldtext.split('\n\n',1)[0]+'\n\n'+notice+oldtext.split('\n\n',1)[1])
print('merged initial brief, current records and detailed report summary')
