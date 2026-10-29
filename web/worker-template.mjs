// 書齋之門與遠方問答，鑰不出後臺。
const shuzhai = __SCROLL__;
const reply = (value, status = 200) => Response.json(value, {status, headers: {'Cache-Control':'no-store'}});
export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if (request.method === 'GET' && path === '/') return new Response(shuzhai.html, {
      headers:{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store',
        'X-Content-Type-Options':'nosniff','Referrer-Policy':'same-origin'}});
    if (request.method === 'GET' && path === '/api/chat-config') {
      if (!env.CUSTOMER_HTTP_MODEL) return reply({cutoff_year:1899,has_library:false,model_ready:false});
      try {const r=await fetch(new URL('/api/chat-config',env.CUSTOMER_HTTP_MODEL),{signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error();return reply(await r.json());}
      catch{return reply({cutoff_year:1899,has_library:false,model_ready:false});}
    }
    if (path !== '/v1/responses') return reply({detail:'此頁未備。'},404);
    if (request.method !== 'POST') return reply({detail:'須呈問始能作答。'},405);
    if (!env.CUSTOMER_HTTP_MODEL) return reply({detail:'自己的模型服務尚未連線。請先啟動 gpt-oss 並設定私人連線。'},503);
    if(request.headers.get('Origin') && request.headers.get('Origin') !== new URL(request.url).origin) return reply({detail:'請由書齋本頁呈問。'},403);
    try {
      const raw = await request.text();
      if(raw.length > 200000) return reply({detail:'此席過長，請另啟一席。'},413);
      const body = JSON.parse(raw);
      let messages = typeof body.input === 'string' ? [{role:'user',content:body.input}] : body.input;
      if(!Array.isArray(messages) || !messages.length || messages.length > 100 || messages.some(m=>!m || !['user','assistant'].includes(m.role) || typeof m.content !== 'string' || !m.content.trim() || m.content.length>100000) || messages.at(-1).role!=='user') return reply({detail:'問答格式不合，請另啟一席再試。'},422);
      if(messages.reduce((sum,m)=>sum+m.content.length,0)>100000) return reply({detail:'此席文字過長，請另啟一席。'},413);
      if(body.stream === true) return reply({detail:'此門僅受完整問答。'},422);
      const tokens=body.max_output_tokens ?? 12000;
      if(!Number.isInteger(tokens)||tokens<256||tokens>16384) return reply({detail:'答覆篇幅須在二五六至一六三八四符之間。'},422);
      messages=messages.map(({role,content})=>({role,content}));
      const upstream=await fetch(new URL('/v1/responses',env.CUSTOMER_HTTP_MODEL),{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({input:messages,max_output_tokens:tokens,stream:false,store:false}),signal:AbortSignal.timeout(180000)});
      if(!upstream.ok) return reply({detail:'模型服務未能作答，請確認模型已載入。'},502);
      const result=await upstream.json();
      return reply({id:result.id,status:result.status,output:(result.output||[]).filter(i=>i.type==='message'),usage:result.usage,error:result.error?{message:'此次未得有效答覆。'}:null});
    } catch(error) {
      if(error instanceof SyntaxError) return reply({detail:'問答格式不合，請再呈問。'},422);
      return reply({detail:'模型服務連線中斷或逾時，請確認模型正在運行。'},504);
    }
  }
};
