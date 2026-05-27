-- Run in Supabase SQL Editor (once per project)

create table if not exists public.app_users (
  id uuid primary key,
  email text not null,
  updated_at timestamptz not null default now()
);

create table if not exists public.scan_history (
  id text primary key,
  user_id uuid not null references public.app_users(id) on delete cascade,
  code text not null,
  language text not null,
  problem_id text null,
  detection_mode text not null,
  risk_score double precision not null,
  decision text not null,
  component_scores jsonb not null default '{}'::jsonb,
  signals jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_scan_history_user_created
  on public.scan_history(user_id, created_at desc);
