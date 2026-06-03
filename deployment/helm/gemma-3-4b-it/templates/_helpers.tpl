gotemplate
{{- define "scalarlm.fullname" -}}
{{- printf "%s" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.vllmname" -}}
{{- printf "%s-vllm" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.megatronname" -}}
{{- printf "%s-megatron" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.cloudflaredFullName" -}}
{{- printf "%s-cloudflared" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "scalarlm.labels" -}}
app.kubernetes.io/name: {{ include "scalarlm.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "scalarlm.vllmlabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.vllmname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "scalarlm.megatronlabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.megatronname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "scalarlm.cloudflaredLabels" -}}
app.kubernetes.io/name: {{ include "scalarlm.cloudflaredFullName" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "scalarlm.scheduling" -}}
{{- $na := .Values.nodeAffinity -}}
{{- if and $na $na.hostnames (not (empty $na.hostnames)) }}
affinity:
  nodeAffinity:
    {{- if $na.preferred }}
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: {{ $na.weight | default 100 }}
        preference:
          matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values:
              {{- range $na.hostnames }}
                - {{ . | quote }}
              {{- end }}
    {{- else }}
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values:
              {{- range $na.hostnames }}
                - {{ . | quote }}
              {{- end }}
    {{- end }}
{{- else if and $na $na.labels (not (empty $na.labels)) }}
affinity:
  nodeAffinity:
    {{- if $na.preferred }}
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: {{ $na.weight | default 100 }}
        preference:
          matchExpressions:
          {{- range $key, $val := $na.labels }}
            - key: {{ $key | quote }}
              operator: In
              values:
                - {{ $val | quote }}
          {{- end }}
    {{- else }}
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
          {{- range $key, $val := $na.labels }}
            - key: {{ $key | quote }}
              operator: In
              values:
                - {{ $val | quote }}
          {{- end }}
    {{- end }}
{{- else if not (empty .Values.affinity) }}
affinity:
  {{- toYaml .Values.affinity | nindent 2 }}
{{- end }}
{{- with .Values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}
