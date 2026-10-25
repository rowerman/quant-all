// 单次SSE请求，不重连；fetch使HTTP错误（含429）可以直接展示。
export async function consumeStream(url, signal, onEvent){
  const response=await fetch(url,{signal});
  if(!response.ok){const error=await response.json();throw new Error(`${error.message} [${error.code}] 请求 ${error.request_id}`);}
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',terminal=false;
  try{
    while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value,{stream:!done});
      let boundary;while((boundary=buffer.indexOf('\n\n'))>=0){const block=buffer.slice(0,boundary);buffer=buffer.slice(boundary+2);let stage='',lines=[];
        for(const line of block.split('\n')){if(line.startsWith('event:'))stage=line.slice(6).trim();else if(line.startsWith('data:'))lines.push(line.slice(5).trimStart());}
        if(stage&&lines.length){onEvent(stage,JSON.parse(lines.join('\n')));if(stage==='done'||stage==='error')terminal=true;}
      }
      if(done)break;
    }
    if(!terminal)throw new Error('连接提前结束，请重新发起计算');
  }finally{reader.releaseLock();}
}
