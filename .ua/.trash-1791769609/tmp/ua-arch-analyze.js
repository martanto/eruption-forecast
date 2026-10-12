const fs=require('fs');const d=JSON.parse(fs.readFileSync(process.argv[2]));
const grp=p=>{const m=p.match(/^src\/eruption_forecast\/([^/]+)\//);if(m)return m[1];if(p.startsWith('src/eruption_forecast/'))return 'pkg-root';if(p.startsWith('wiki/'))return 'wiki';return 'root';};
const g={};const id2g={};for(const n of d.fileNodes){const k=grp(n.filePath);(g[k]=g[k]||[]).push(n.id);id2g[n.id]=k;}
const inter={},intra={},fi={},fo={};
for(const e of d.importEdges){const a=id2g[e.source],b=id2g[e.target];if(!a||!b)continue;fo[e.source]=(fo[e.source]||0)+1;fi[e.target]=(fi[e.target]||0)+1;
 if(a===b){intra[a]=(intra[a]||0)+1}else{const k=a+'->'+b;inter[k]=(inter[k]||0)+1}}
const dir=[];for(const k in inter){const[a,b]=k.split('->');const r=inter[b+'->'+a]||0;if(inter[k]>r)dir.push({dependent:a,dependsOn:b,count:inter[k],reverse:r})}
const types={};for(const n of d.fileNodes)types[n.type]=(types[n.type]||0)+1;
fs.writeFileSync(process.argv[3],JSON.stringify({scriptCompleted:true,directoryGroups:g,nodeTypeCounts:types,totalFileNodes:d.fileNodes.length,
 interGroupImports:Object.entries(inter).map(([k,c])=>({from:k.split('->')[0],to:k.split('->')[1],count:c})).sort((x,y)=>y.count-x.count),
 intraGroup:intra,dependencyDirection:dir,fileFanIn:fi,fileFanOut:fo},null,1));
